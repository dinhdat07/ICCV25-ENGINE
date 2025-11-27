import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinear_RanPAC
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
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
        self.image_adapter = None
        self.text_adapter = None
        self.beta = 1
        self.decay = 1
        self.mix_b = get_attribute(args, "mix_bias", 0.6)
        self.cov_shrink = get_attribute(args, "cov_shrink", False)
        
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_edge_distance = []
        self.old_image_adapter = None
        self.old_text_adapter = None
        
    def update_stat(self, known_classes, total_classes, train_loader,device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            # vecs_512 = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                # image_features_512 = image_features @ self.visual_proj
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                vecs.append(image_features)
                # vecs_512.append(image_features_512)
                labels.append(targets)

            vecs = torch.cat(vecs)
            # vecs_512 = torch.cat(vecs_512)
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

    def update_task(self):
        from convs.linears import Adapter,MLP_Adapter
        device = next(self.model.parameters()).device
        if self.image_adapter is not None:
            self.old_image_adapter = copy.deepcopy(self.image_adapter)
            self.old_text_adapter = copy.deepcopy(self.text_adapter)
            self.freeze(self.old_image_adapter)
            self.freeze(self.old_text_adapter)
        else:
            self.old_image_adapter = None
            self.old_text_adapter = None
            self.image_adapter = MLP_Adapter(512, 512).to(device)
            self.text_adapter = MLP_Adapter(512, 512).to(device)
    
    def Image_encode(self, image_features):
        if self.image_adapter is None:
            return image_features
        return self.image_adapter(image_features)
    
    def Text_encode(self, text_features):
        if self.text_adapter is None:
            return text_features
        return self.text_adapter(text_features)
    
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

    def analyze_mean_cov(self, features, labels):
        label_ids = torch.sort(torch.unique(labels))[0]
        for l in label_ids:
            index = torch.nonzero(labels == l).squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append(
                (max_distance.mean() - max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean())
            )
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / (mask.sum() + 1e-12)
        iden = torch.eye(cov.shape[0], device=cov.device)
        alpha1 = 1
        alpha2 = 1
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        return cov_

    def sample(self, mean, cov, size, shrink=False):
        vec = torch.randn(size, mean.shape[-1], device=mean.device)
        cov_mat = self.shrink_cov(cov) if shrink else cov
        sqrt_cov = torch.linalg.cholesky(cov_mat)
        vec = vec @ sqrt_cov.t()
        vec = vec + mean
        return vec

    def sample_class(self, class_idx: int, size: int):
        if class_idx >= len(self.class_mean_list) or class_idx >= len(self.class_cov_list):
            return None
        return self.sample(self.class_mean_list[class_idx], self.class_cov_list[class_idx], size, shrink=self.cov_shrink)
        
    def freeze(self, model):
        if model is None:
            return
        for param in model.parameters():
            param.requires_grad = False
            
    def activate_old_adapter(self):
        if self.image_adapter is not None:
            for param in self.image_adapter.parameters():
                param.requires_grad = True
        if self.text_adapter is not None:
            for param in self.text_adapter.parameters():
                param.requires_grad = True

    def _get_first_linear(self, adapter):
        if adapter is None:
            return None
        for module in adapter.modules():
            if isinstance(module, nn.Linear):
                return module
        return None

    def _mix_single_adapter(self, new_adapter, old_adapter):
        new_linear = self._get_first_linear(new_adapter)
        old_linear = self._get_first_linear(old_adapter)
        if new_linear is None or old_linear is None:
            return
        weight_new = new_linear.weight.data
        weight_old = old_linear.weight.data
        U_old, S_old, V_old = torch.linalg.svd(weight_old, full_matrices=False)
        P_new = U_old.T @ weight_new
        baseline = torch.diag(S_old) @ V_old
        dist = (P_new - baseline).abs()
        denom = torch.clamp(dist.max(), min=1e-12)
        mask = torch.clamp(dist / denom + self.mix_b, max=1)
        right = P_new * mask + baseline * (1 - mask)
        mixed_weight = U_old @ right
        new_linear.weight.data.copy_(mixed_weight)
        if new_linear.bias is not None and old_linear.bias is not None:
            new_linear.bias.data.copy_((1 - self.mix_b) * new_linear.bias.data + self.mix_b * old_linear.bias.data)

    def mix_matrix(self):
        self._mix_single_adapter(self.image_adapter, self.old_image_adapter)
        self._mix_single_adapter(self.text_adapter, self.old_text_adapter)
        self.old_image_adapter = None
        self.old_text_adapter = None
