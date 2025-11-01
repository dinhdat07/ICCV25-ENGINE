import copy
from itertools import chain
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinear_RanPAC
import timm
import torch.nn.functional as F
from convs.projections import ENGINE_Adapter, Proj_Pure_MLP, MultiHeadAttention
import os
import json
import torchvision.transforms as transforms
from utils.toolkit import get_attribute
import difflib
from PIL import Image
import random
random.seed(1993)

def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()

    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )



class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out



class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name='SimpleClipNet'
        self.args=args


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):

        image_features, text_features, logit_scale=self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)

class Engine(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        self.args=args
        self.freeze(self.model)
        self.image_adapters =  nn.ModuleList()
        self.text_adapters = nn.ModuleList()

        # TUNA: non-linear adapter
        dropout_rate = float(getattr(args, 'dropout', 0.1))
        self.uni_image_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.uni_text_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.freeze(self.uni_image_adapter)
        self.freeze(self.uni_text_adapter)

        self.image_fusion_alpha = nn.Parameter(torch.ones(1)) 
        self.image_fusion_beta = nn.Parameter(torch.ones(1)) 
        
        self.text_fusion_alpha = nn.Parameter(torch.ones(1))
        self.text_fusion_beta = nn.Parameter(torch.ones(1))

        # RAPF: mix matrix parameters
        self.beta = 1
        self.decay = 1
        self.mix_b = get_attribute(args, "mix_bias", 0.6)
        
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_edge_distance = []

        self.class_name_features = None
        self.hard_pairs = None
                        
    def update_stat(self, known_classes, total_classes, train_loader,device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                vecs.append(image_features)
                labels.append(targets)

            vecs = torch.cat(vecs)
            labels = torch.cat(labels)
            
            mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
            center_vecs = torch.cat([vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
            cov_inv = center_vecs.T @ center_vecs /(center_vecs.shape[0] - 1)
            cov_inv =  center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())    
            if not hasattr(self, 'mu'):
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (known_classes/total_classes)*self.cov_inv + (total_classes-known_classes)/total_classes*cov_inv + ((known_classes/total_classes)*(total_classes-known_classes)/total_classes**2)*(self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
                self.mu = torch.cat([self.mu, mu])
            ps = torch.ones(self.mu.shape[0]).cuda() * 1. / self.mu.shape[0]
            self.W = torch.einsum('nd, dc -> cn', self.mu, self.cov_inv)
            self.b = ps.log() - torch.einsum('nd, dc, nc -> n', self.mu, self.cov_inv, self.mu) / 2

    def prepare_task(self, class_names, templates, known_classes, total_classes, threshold, device):
        self.hard_pairs = None
        if total_classes == 0:
            return

        prompts = []
        for cname in class_names:
            for tmpl in templates:
                prompts.append(tmpl.format(cname))
        tokens = self.tokenizer(prompts).to(device)
        text_features = self.model.encode_text(tokens)
        templates_per_class = len(templates)
        if templates_per_class > 0:
            text_features = text_features.view(total_classes, templates_per_class, -1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.class_name_features = text_features

        if known_classes <= 0 or total_classes <= known_classes or threshold is None:
            return

        old_features = self.class_name_features[:known_classes].type(torch.float32)
        new_features = self.class_name_features[known_classes:total_classes].type(torch.float32)
        dist = torch.cdist(old_features, new_features)
        indices = torch.nonzero(dist < threshold, as_tuple=False)
        if indices.numel() == 0:
            return
        pairs = indices.clone()
        pairs[:, 1] += known_classes
        self.hard_pairs = pairs

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        if self.visual.attn_pool is not None:
            if self.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.visual.attn_pool(x)
                if self.visual.attn_pool_type == 'parallel':
                    pooled = self.visual.attn_pool_contrastive(x)
                else:
                    assert self.visual.attn_pool_type == 'cascade'
                    pooled = self.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.visual.attn_pool(x)
                x = self.visual.ln_post(x)
                pooled, tokens = self.visual._global_pool(x)
        elif self.visual.final_ln_after_pool:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        else:
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        return pooled
    
    def get_trainable_parameters(self):
        params = []
        if self.image_adapters and len(self.image_adapters) > 0:
            params.append(self.image_adapters[-1].parameters())
            params.append([self.image_fusion_alpha, self.image_fusion_beta])
        if self.text_adapters and len(self.text_adapters) > 0:
            params.append(self.text_adapters[-1].parameters())
            params.append([self.text_fusion_alpha, self.text_fusion_beta])
        return chain.from_iterable(params)

    def update_task(self, noise_std: float = 0.01):
        
        for inj in self.image_adapters: 
            self.freeze(inj)
        for inj in self.text_adapters:
            self.freeze(inj)

        dropout_rate = float(getattr(self.args, 'dropout', 0.1))

        # image adapter
        if self.image_adapters:
            new_image_adapter = copy.deepcopy(self.image_adapters[-1])
            for param in new_image_adapter.parameters():
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True 
            
            if hasattr(new_image_adapter, 'scale'):
                with torch.no_grad():
                    new_image_adapter.scale.fill_(1.0)
            self.image_adapters.append(new_image_adapter.to(self.device).to(dtype=self.dtype))
        else:
            self.image_adapters.append(
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )

        # text adapter
        if self.text_adapters:
            new_text_adapter = copy.deepcopy(self.text_adapter[-1])
            for param in new_text_adapter.parameters():
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True
            if hasattr(new_text_adapter, 'scale'):
                with torch.no_grad():
                    new_text_adapter.scale.fill_(1.0)
            self.text_adapters.append(new_text_adapter.to(self.device).to(dtype=self.dtype))
        else:
            self.text_adapters.append(
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )
        
        # fusion for universal adapter
        # if len(self.image_adapters) > 1:
        #     self.mix_matrix()

    def _flatten_adapter_params(self, adapter):
        params = []
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                params.append(param.data.flatten())
        return torch.cat(params)

    def _unflatten_adapter_params(self, adapter, flat_vector: torch.Tensor):
        pointer = 0
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                num_elements = param.numel()
                param.data.copy_(
                    flat_vector[pointer:pointer + num_elements].view_as(param.data)
                )
                pointer += num_elements
        return adapter
    
    def mix_matrix(self):
        if len(self.image_adapters) < 1:
            return 
            
        all_flat_vectors = []
        for adapter in self.image_adapters: 
            all_flat_vectors.append(self._flatten_adapter_params(adapter))
        v_uni_img = torch.stack(all_flat_vectors).mean(dim=0)
        self._unflatten_adapter_params(self.uni_image_adapter, v_uni_img)
        self.freeze(self.uni_image_adapter) 

        all_flat_vectors = []
        for adapter in self.text_adapters:
            all_flat_vectors.append(self._flatten_adapter_params(adapter))
        v_uni_txt = torch.stack(all_flat_vectors).mean(dim=0)
        self._unflatten_adapter_params(self.uni_text_adapter, v_uni_txt)
        self.freeze(self.uni_text_adapter)
            
    def Text_encode(self, features: torch.Tensor):
        if len(self.text_adapters) == 0:
            return features
                    
        device = features.device
        
        try:
            target_dtype = next(self.text_adapters[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        uni_output = self.uni_text_adapter(features)
        task_output = self.text_adapters[-1](features)

        alpha = self.text_fusion_alpha.to(device)
        beta = self.text_fusion_beta.to(device)

        fusion_weights = torch.stack([alpha, beta], dim=0)
        normalized_weights = F.softmax(fusion_weights, dim=0)
        alpha_hat, beta_hat = normalized_weights[0], normalized_weights[1]

        outputs = (alpha_hat * uni_output) + (beta_hat * task_output)
                    
        return outputs
    
    def Image_encode(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.image_adapters) == 0:
            return features
        device = features.device 
        try:
            target_dtype = next(self.image_adapters[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        uni_output = self.uni_image_adapter(features)
        task_output = self.image_adapters[-1](features)

        alpha = self.image_fusion_alpha.to(device)
        beta = self.image_fusion_beta.to(device)
        
        fusion_weights = torch.stack([alpha, beta], dim=0)
        normalized_weights = F.softmax(fusion_weights, dim=0)
        alpha_hat, beta_hat = normalized_weights[0], normalized_weights[1]

        outputs = (alpha_hat * uni_output) + (beta_hat * task_output)
        
        return outputs
    
    @property
    def feature_dim(self):
        return self.model.out_dim
    
    def extract_vector(self, x):
        return self.model.encode_image(x)

    def encode_image(self, x):
        imag_features =  self.model.encode_image(x)
        imag_res = self.Image_encode(imag_features)
        return imag_res
    
    def encode_text(self, x):
        text_features = self.model.encode_text(x)
        text_res = self.Text_encode(text_features)
        return text_res

    def forward(self, img, text):
        image_features, text_features, logit_scale=self.model(img, text)
        return image_features, text_features, logit_scale

    def rerank(self, des_dict, outputs, image_features_raw, class_to_label, device, topk=5):
        with torch.no_grad():
            top5_predict = outputs.topk(topk, 1, True, True)[1]
            top5_predict_labels = [[class_to_label[int(label)] for label in pred] for pred in top5_predict]
            logi = 0
            for _ in range(3):
                texts = []
                for batch in range(image_features_raw.shape[0]):
                    for main_label in top5_predict_labels[batch]:
                        for second_label in top5_predict_labels[batch]:
                            if main_label == second_label:
                                continue
                            texts.append(main_label + ' with ' + random.choice(des_dict[main_label][second_label]).lower())
                texts = self.tokenizer(texts).to(device)
                texts = self.model.encode_text(texts)
                texts = texts.reshape(image_features_raw.shape[0], topk, topk-1, -1)
                texts = torch.mean(texts, dim=2)
                texts = texts / texts.norm(dim=-1, keepdim=True)
                logits = [image_features_raw[i] @ texts[i].T for i in range(image_features_raw.shape[0])]
                logits = torch.stack(logits)
                logi += logits
            logits = logi/3
            new_logits = torch.zeros_like(outputs)
            for i in range(image_features_raw.shape[0]):
                new_logits[i, top5_predict[i]] = logits[i]
            return new_logits
        
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def analyze_mean_cov(self, features, labels):
        if features.numel() == 0:
            return
        unique_labels = torch.sort(torch.unique(labels))[0]
        for l in unique_labels:
            idx = torch.nonzero(labels == l, as_tuple=False).squeeze()
            class_data = features[idx]
            if class_data.ndim == 1:
                class_data = class_data.unsqueeze(0)
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            stats = (
                max_distance.mean() - max_distance.min(),
                max_distance.max() - max_distance.mean(),
                max_distance.mean(),
            )
            class_idx = int(l.item())
            while len(self.class_mean_list) <= class_idx:
                self.class_mean_list.append(None)
                self.class_cov_list.append(None)
                self.class_edge_distance.append(None)
            self.class_mean_list[class_idx] = mean
            self.class_cov_list[class_idx] = cov
            self.class_edge_distance[class_idx] = stats