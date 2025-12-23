import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.migration_attention = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.migration_gate = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.Sigmoid()
        )

        self.migration_cache = {}  # 缓存历史迁移记录
        self.cache_timestamp = -1  # 缓存的时间戳

        # self.migration_lstm = nn.LSTM(
        #     input_size=h_dim * 2,  # 修改为：实体+关系
        #     hidden_size=h_dim,
        #     num_layers=1,
        #     batch_first=True
        # )

        # 替换LSTM为Transformer
        self.migration_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim * 2,  # 输入维度：实体+关系
                nhead=4,           # 注意力头数
                dim_feedforward=h_dim * 4,  # 前馈网络维度
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2           # Transformer层数
        )
        
        self.migration_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_normal_(self.migration_weight)
        
        self.migration_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.migration_bias)

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def forward(self, g_list, static_graph, use_cuda, migration_sequences=None):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)

        # migration_embs = []
        # if migration_sequences is not None:
        #     for seq in migration_sequences:
        #         # 将序列转换为嵌入
        #         seq_emb = self._process_migration_sequence(seq, self.h, self.emb_rel)
        #         migration_embs.append(seq_emb)
        
        # 处理迁移序列（批处理版本）
        migration_embs = []
        if migration_sequences is not None and len(migration_sequences) > 0:
            migration_embs = self._process_migration_sequence(migration_sequences, self.h, self.emb_rel)
    
        
        return history_embs, static_emb, self.h_0, gate_list, degree_list, migration_embs
        # return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, use_cuda, migration_sequences=None):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _, migration_embs = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        loss_migration = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        
        if migration_sequences is not None:
             # # # # # # # # # # # # # 使用注意力机制融合迁移嵌入和实体嵌入 # # # # # # # # # # # # # 
            for i, (mig_emb, triple) in enumerate(zip(migration_embs, triples)):
                entity_id = triple[0]
                entity_emb = pre_emb[entity_id].unsqueeze(0).unsqueeze(0)  # [1, 1, h_dim]
                mig_emb = mig_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, h_dim]
                
                # 注意力融合
                attended_emb, attention_weights = self.migration_attention(
                    query=entity_emb,
                    key=mig_emb,
                    value=mig_emb
                )
                
                # 门控机制融合
                gate_input = torch.cat([entity_emb.squeeze(0), attended_emb.squeeze(0)], dim=-1)
                gate = self.migration_gate(gate_input)
                
                fused_emb = gate * entity_emb.squeeze(0) + (1 - gate) * attended_emb.squeeze(0)
                
                # 计算重构损失
                reconstruction_loss = F.mse_loss(fused_emb, entity_emb.squeeze(0))
                loss_migration += reconstruction_loss
        return loss_ent, loss_rel, loss_static, loss_migration

        # # # # # # # # # # # # # 使用余弦相似度进行约束 # # # # # # # # # # # # # 
        #     # _, _, _, _, _, migration_embs = self.forward(glist, static_graph, use_cuda, migration_sequences)
            
        #     # 计算迁移嵌入与实体嵌入的相似度约束
        #     for i, (mig_emb, triple) in enumerate(zip(migration_embs, triples)):
        #         entity_id = triple[0]
        #         entity_emb = pre_emb[entity_id]
                
        #         # 移除mig_emb的批次维度
        #         mig_emb = mig_emb.squeeze(0)  # 从 [1, h_dim] 变为 [h_dim]
                
        #         # 余弦相似度约束
        #         similarity = F.cosine_similarity(mig_emb, entity_emb, dim=0)
        #         loss_migration += (1 - similarity)  # 最大化相似度  # 最大化相似度

        # return loss_ent, loss_rel, loss_static, loss_migration
        # # return loss_ent, loss_rel, loss_static

    def build_migration_sequence(self, entity_id, relation_id, target_id, timestamp, history_data):
        """
        构建历史迁移序列，使用缓存优化
        """
        if timestamp != self.cache_timestamp:
            self._update_migration_cache(history_data, timestamp)
        
        cache_key = (entity_id, relation_id)
        if cache_key in self.migration_cache:
            historical_migrations = self.migration_cache[cache_key]
        else:
            historical_migrations = []
        
        import heapq
        heap = []
        
        for target, time_val in historical_migrations:
            if len(heap) < 10:
                heapq.heappush(heap, (-time_val, target))
            else:
                if time_val > -heap[0][0]:
                    heapq.heapreplace(heap, (-time_val, target))
        
        migration_sequence = []
        historical_migrations_sorted = []
        while heap:
            neg_time, target = heapq.heappop(heap)
            historical_migrations_sorted.append((target, -neg_time))
        
        historical_migrations_sorted.sort(key=lambda x: x[1], reverse=True)
        
        for target, time_val in historical_migrations_sorted:
            migration_sequence.extend([entity_id, relation_id, time_val, target])
        
        migration_sequence.extend([entity_id, relation_id, timestamp, target_id])
        
        return torch.tensor(migration_sequence, dtype=torch.long)

    def _process_migration_sequence(self, sequences, entity_emb, rel_emb):
        """
        批处理迁移序列并使用LSTM获得嵌入
        sequences: 迁移序列列表，每个序列是一维张量
        """
        if not sequences:
            return []
        
        device = entity_emb.device
        
        sequences = [seq.to(device) for seq in sequences]
        
        max_seq_length = max(len(seq) // 4 for seq in sequences)
        
        batch_size = len(sequences)
        batch_combined = []
        
        for seq in sequences:
            seq_length = len(seq) // 4
            sequence_2d = seq.view(seq_length, 4)
            
            entities = sequence_2d[:, 0]  # 实体ID
            relations = sequence_2d[:, 1]  # 关系ID
            
            entity_embs = entity_emb[entities]
            rel_embs = rel_emb[relations]
            
            combined = torch.cat([entity_embs, rel_embs], dim=-1)  # [seq_length, h_dim*2]
            
            if seq_length < max_seq_length:
                padding = torch.zeros(max_seq_length - seq_length, combined.size(1), 
                                    device=device, dtype=combined.dtype)
                combined = torch.cat([combined, padding], dim=0)
            
            batch_combined.append(combined)
        
        batch_combined = torch.stack(batch_combined, dim=0)
        
        seq_lengths = [len(seq) // 4 for seq in sequences]
        mask = torch.arange(max_seq_length, device=device).expand(batch_size, max_seq_length) < torch.tensor(seq_lengths, device=device).unsqueeze(1)
        
        # packed_input = nn.utils.rnn.pack_padded_sequence(
        #     batch_combined, 
        #     seq_lengths, 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )

        # packed_output, (hidden, _) = self.migration_lstm(packed_input)
        
        # migration_embs = hidden.squeeze(0)  # [batch_size, h_dim]

        src_key_padding_mask = mask  # 填充位置的掩码
        
        transformer_output = self.migration_transformer(
            batch_combined,
            src_key_padding_mask=src_key_padding_mask
        )

        migration_embs = []
        for i in range(batch_size):
            # 获取最后一个非填充位置的输出
            last_valid_idx = seq_lengths[i] - 1
            migration_embs.append(transformer_output[i, last_valid_idx])
        
        migration_embs = torch.stack(migration_embs, dim=0) 

        return migration_embs
    
    #     """处理迁移序列并使用LSTM获得嵌入"""
    #     # sequence是一维张量，格式为[实体, 关系, 时间, 目标实体, 实体, 关系, 时间, 目标实体, ...]
        
    #     device = entity_emb.device
    #     sequence = sequence.to(device)
        
    #     seq_length = len(sequence) // 4
    #     sequence_2d = sequence.view(seq_length, 4)
        
    #     entities = sequence_2d[:, 0]  # 实体ID
    #     relations = sequence_2d[:, 1]  # 关系ID  
    #     times = sequence_2d[:, 2]  # 时间戳
        
    #     entity_embs = entity_emb[entities]
    #     rel_embs = rel_emb[relations]
        
    #     combined = torch.cat([entity_embs, rel_embs], dim=-1)
        
    #     combined = combined.unsqueeze(0)  # [1, seq_length, feature_dim]
        
    #     lstm_out, _ = self.migration_lstm(combined)
        
    #     migration_emb = lstm_out[:, -1, :].squeeze(0)
        
    #     return migration_emb
    def _update_migration_cache(self, history_data, current_timestamp):
        """
        更新迁移缓存，只在时间戳变化时调用
        """
        self.migration_cache = {}
        
        # 遍历所有历史快照，构建缓存
        for time_idx, time_slice in enumerate(history_data):
            for triple in time_slice:
                entity_id, relation_id, target_id = triple[0], triple[1], triple[2]
                cache_key = (entity_id, relation_id)
                
                if cache_key not in self.migration_cache:
                    self.migration_cache[cache_key] = []
                
                # 添加迁移记录
                self.migration_cache[cache_key].append((target_id, time_idx))
        
        self.cache_timestamp = current_timestamp