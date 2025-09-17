import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def IMAL(image_features, text_features, logit_scale=1.0, temperature=0.5, gamma=0.05, kl_weight=0.005, epsilon=1e-8, device='cuda'):
    """
    
    参数:
        image_features (torch.Tensor): 图像特征，形状 [batch_size, feature_dim]
        text_features (torch.Tensor): 文本特征，形状 [batch_size, feature_dim]
        logit_scale (float): 内部模态损失的缩放因子
        temperature (float): 对比损失的温度参数
        gamma (float): 内部模态损失的权重
        kl_weight (float): KL 散度的权重
        epsilon (float): 数值稳定性参数
        device (str): 设备 ('cuda' 或 'cpu')
    """
    batch_size = image_features.size(0)
    
    # 归一化特征
    image_norm = F.normalize(image_features, dim=-1, eps=epsilon)
    text_norm = F.normalize(text_features, dim=-1, eps=epsilon)
    
    # 计算跨模态相似性矩阵
    similarity = torch.clamp(image_norm @ text_norm.T, min=-10.0, max=10.0)  # 限制范围
    scores = torch.exp(torch.clamp(similarity / temperature, min=-10.0, max=10.0))  # 限制指数输入
    
    # 计算对角线（正样本对）
    diagonal = scores.diag().view(-1, 1).to(device)
    
    # 计算行和列的总和（所有样本对）
    sum_row = scores.sum(dim=1) + epsilon  # 防止除以零
    sum_col = scores.sum(dim=0) + epsilon  # 防止除以零
    
    # 跨模态损失 (InfoNCE 风格)
    loss_cm = -torch.log((diagonal + epsilon) / sum_row).mean()
    
    # 内部模态损失
    intra_image_norm = F.normalize(image_features, dim=-1, eps=epsilon)
    intra_text_norm = F.normalize(text_features, dim=-1, eps=epsilon)
    logits_topo = logit_scale * (intra_image_norm @ intra_text_norm.T)
    pos_topo = torch.exp(torch.clamp(torch.diag(logits_topo), min=-10.0, max=10.0)) + epsilon
    neg_topo = torch.exp(torch.clamp(logits_topo, min=-10.0, max=10.0)).sum(dim=1) + epsilon
    loss_im = -torch.log(pos_topo / neg_topo).mean()
    
    # 总对比损失
    contrastive_loss = loss_cm + gamma * loss_im
    
    # KL 散度正则化
    # 使用 softmax 后的行和列分布作为模型学习的分布
    p_row = scores / sum_row.view(-1, 1)  # 形状 [batch_size, batch_size]
    p_col = scores.t() / sum_col.view(-1, 1)  # 转置后归一化
    
    # 均匀分布作为先验 (每个样本对的概率为 1/batch_size^2)
    uniform_dist = torch.ones_like(p_row) / (batch_size * batch_size + epsilon)
    
    # 计算 KL 散度 (对每个样本的分布求平均)
    kl_row = F.kl_div(torch.log(p_row + epsilon), uniform_dist, reduction='none').sum(dim=1).mean()
    kl_col = F.kl_div(torch.log(p_col + epsilon), uniform_dist, reduction='none').sum(dim=1).mean()
    kl_loss = (kl_row + kl_col) / 2
    
    # 总损失，添加缩放因子以减小损失值
    total_loss = (contrastive_loss + kl_weight * kl_loss)
    
    # 确保返回标量
    # total_loss = total_loss.mean()
    # print("total_loss:", total_loss)
    
    return total_loss