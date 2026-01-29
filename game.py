import numpy as np
import torch
import time
import os  # [New] 用于文件操作
from tqdm import tqdm

from config import Config
from world import KoanAtlas
from physics import IsingModel
from dynamics import FastSolver, DPP, SlowLearner

class RuleEngine:
    """
    规则引擎：定义 9 局游戏的 Ground Truth 逻辑。
    负责判断某个公案是否符合特定规则。
    """
    def __init__(self, atlas):
        self.atlas = atlas
        self.rules = [
            ("存在红色物体", self.rule_exist_red),
            ("尺寸完全一致", self.rule_same_size),
            ("蓝色物体唯一", self.rule_unique_blue),
            ("存在蓝色小物体", self.rule_exist_blue_small),
            ("全员蓝色或小尺寸", self.rule_all_blue_or_small),
            ("红者最大", self.rule_red_is_largest),
            ("物体间有接触", self.rule_contact_exists),
            ("红蓝相触", self.rule_red_blue_contact),
            ("存在堆叠现象", self.rule_stack_exists)
        ]

    def get_ground_truth_vector(self, rule_idx):
        """计算全宇宙 5127 个公案在指定规则下的真值向量 (+1/-1)"""
        rule_name, rule_func = self.rules[rule_idx]
        print(f"\n📜 正在编译规则 [{rule_idx+1}/9]: {rule_name} ...")
        
        gt_vector = np.zeros(self.atlas.num_koans, dtype=np.float32)
        
        for i in range(self.atlas.num_koans):
            is_true = rule_func(i)
            gt_vector[i] = 1.0 if is_true else -1.0
            
        return gt_vector, rule_name

    # --- 辅助提取函数 ---
    def _get_active_feats(self, idx):
        mass = self.atlas.mass_tensor[idx]
        active = np.where(mass > 0)[0]
        feats = self.atlas.feature_tensor[idx][active]
        return feats, active

    def _get_colors(self, feats):
        # one-hot argmax: 0=R, 1=G, 2=B
        return np.argmax(feats[:, Config.IDX_COLOR], axis=1)

    def _get_sizes(self, feats):
        # one-hot argmax: 0=S, 1=M, 2=L
        return np.argmax(feats[:, Config.IDX_SIZE], axis=1)
    
    def _get_grounds(self, feats):
        # scalar: 0=Fly, 1=Ground
        return feats[:, Config.IDX_GROUND].flatten()

    # --- 具体规则实现 ---

    def rule_exist_red(self, idx):
        feats, _ = self._get_active_feats(idx)
        colors = self._get_colors(feats)
        return 0 in colors  # 0 is Red

    def rule_same_size(self, idx):
        feats, _ = self._get_active_feats(idx)
        if len(feats) == 0: return False # 防御性编程
        sizes = self._get_sizes(feats)
        return np.all(sizes == sizes[0])

    def rule_unique_blue(self, idx):
        feats, _ = self._get_active_feats(idx)
        colors = self._get_colors(feats)
        return np.sum(colors == 2) == 1 # 2 is Blue

    def rule_exist_blue_small(self, idx):
        feats, _ = self._get_active_feats(idx)
        colors = self._get_colors(feats)
        sizes = self._get_sizes(feats)
        # Blue is 2, Small is 0
        is_blue = (colors == 2)
        is_small = (sizes == 0)
        return np.any(is_blue & is_small)

    def rule_all_blue_or_small(self, idx):
        feats, _ = self._get_active_feats(idx)
        colors = self._get_colors(feats)
        sizes = self._get_sizes(feats)
        # (Blue OR Small) for ALL
        condition = (colors == 2) | (sizes == 0)
        return np.all(condition)

    def rule_red_is_largest(self, idx):
        # 红者最大：至少有一个红色，且红色尺寸 > 所有非红色尺寸
        feats, _ = self._get_active_feats(idx)
        colors = self._get_colors(feats)
        sizes = self._get_sizes(feats)
        
        red_mask = (colors == 0)
        non_red_mask = ~red_mask
        
        if np.sum(red_mask) == 0: return False # 必须有红色
        
        # [Audit Fix] 逻辑修复：如果全员都是红色，没有非红物体，则“所有非红物体”的条件空缺。
        # 语义上，既然没有非红物体可以比红色大，那么红色就是最大的。
        if np.sum(non_red_mask) == 0: return True
        
        max_red_size = np.max(sizes[red_mask])
        max_non_red_size = np.max(sizes[non_red_mask])
        
        return max_red_size > max_non_red_size

    def rule_contact_exists(self, idx):
        # 邻接矩阵有值
        mass = self.atlas.mass_tensor[idx]
        n = int(np.sum(mass > 0))
        adj = self.atlas.structure_tensor[idx][:n, :n]
        # 邻接矩阵是对称的，对角线为0。只要sum > 0 就说明有边
        return np.sum(adj) > 0

    def rule_red_blue_contact(self, idx):
        feats, active = self._get_active_feats(idx)
        n = len(active)
        adj = self.atlas.structure_tensor[idx][:n, :n]
        colors = self._get_colors(feats) # 0=R, 2=B
        
        rows, cols = np.where(adj > 0)
        for r, c in zip(rows, cols):
            # 检查连接的两个节点的颜色
            c1, c2 = colors[r], colors[c]
            if (c1 == 0 and c2 == 2) or (c1 == 2 and c2 == 0):
                return True
        return False

    def rule_stack_exists(self, idx):
        # 存在堆叠：即存在一个物体没有接地 (Ground=0)
        # 注意：dataset/config 定义 Ground=1 为接地，0 为悬空(堆叠在别人上面)
        feats, _ = self._get_active_feats(idx)
        grounds = self._get_grounds(feats)
        # 如果存在 ground == 0，说明有堆叠
        return np.any(grounds < 0.5)


class ZendoGame:
    def __init__(self):
        # 1. 初始化基础设施
        self.atlas = KoanAtlas()
        self.rule_engine = RuleEngine(self.atlas)
        
        # 2. 核心组件
        self.physics = IsingModel(self.atlas.num_koans)
        self.solver = FastSolver()
        self.dpp = DPP()
        self.learner = SlowLearner(self.atlas)
        
        # 3. [New] 实验记录目录
        self.log_dir = "hypotheses_log"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 已创建实验记录文件夹: {self.log_dir}")

    def play_all_games(self):
        """执行所有 9 局游戏"""
        results = []
        for i in range(9):
            score, rule_name = self.run_single_game(rule_idx=i)
            results.append((rule_name, score))
            
        print("\n" + "="*60)
        print("🏆 Zendo 挑战赛最终成绩单")
        print("="*60)
        avg_score = 0
        for name, score in results:
            print(f"规则: {name:<20} | 准确率: {score*100:.2f}%")
            avg_score += score
        print("-" * 60)
        print(f"平均准确率: {(avg_score/9)*100:.2f}%")
        print("="*60)

    def run_single_game(self, rule_idx):
        # --- Game Setup ---
        gt_vector, rule_name = self.rule_engine.get_ground_truth_vector(rule_idx)
        
        print(f"\n🎮 [GAME {rule_idx+1}] 规则: {rule_name}")
        
        # [New] 为当前游戏创建独立文件夹
        # 处理一下文件名中的特殊字符
        safe_rule_name = rule_name.replace(" ", "_").replace("/", "-")
        game_log_dir = os.path.join(self.log_dir, f"Game_{rule_idx+1}_{safe_rule_name}")
        if not os.path.exists(game_log_dir):
            os.makedirs(game_log_dir)
        
        # 1. Reset State
        # 重置注意力为均匀
        current_attention = Config.INIT_ATTENTION.copy()
        
        # 2. Provide Initial Positive Example (Pinning Field Start)
        # 随机找一个正例
        pos_indices = np.where(gt_vector > 0)[0]
        neg_indices = np.where(gt_vector < 0)[0]
        
        known_pos = [np.random.choice(pos_indices)]
        known_neg = [] # 初始无反例
        
        print(f"   🚩 初始提示 (正例): Koan #{known_pos[0]}")

        # --- Game Loop (7 Rounds) ---
        final_hypotheses = []
        
        for round_num in range(Config.MAX_ROUNDS - 1): # Run 0 to 8, but actually logic says 7 rounds of active test
            if round_num >= 7: break # 题目要求7轮检验

            # A. Update Physics (J & h)
            # 根据当前注意力计算距离，更新 J
            dist_matrix = self.atlas.get_weighted_distance_matrix(current_attention)
            self.physics.update_couplings(dist_matrix)
            
            # 设置钉扎场
            self.physics.set_pinning_field(known_pos, known_neg, round_num)
            
            # B. Fast Dynamics (Generate Hypotheses)
            # 随机初始化自旋
            chains = []
            
            # [Audit Optimization] 使用更强的迭代步数，确保覆盖 5127 个公案
            # Config.MCMC_STEPS (1000) 太小，这里显式 Override 为 System_Size * Sweeps (e.g., 20)
            mcmc_steps_override = self.atlas.num_koans * 20
            
            for _ in range(Config.NUM_CHAINS):
                init_spins = np.random.choice([-1.0, 1.0], size=self.atlas.num_koans)
                final_spins = self.solver.run_glauber(self.physics, init_spins, steps=mcmc_steps_override)
                chains.append(final_spins)
            
            # C. DPP Selection
            # 选出 3 个代表性假设
            selected_indices, _ = self.dpp.select_subset(chains, self.physics)
            current_hypotheses = [chains[i] for i in selected_indices]
            
            # 如果 DPP 选不够3个，补全
            while len(current_hypotheses) < 3:
                current_hypotheses.append(chains[np.random.randint(len(chains))])
            
            # --- [New] 记录本轮假设并计算准确率 ---
            hyp_stack = np.stack(current_hypotheses) # (3, N)
            # 计算准确率: 与 Ground Truth 对比
            # 准确率 = (Prediction == GT) 的平均值
            accuracies = np.mean(hyp_stack == gt_vector, axis=1)
            
            acc_str = " | ".join([f"H{i}: {acc*100:.1f}%" for i, acc in enumerate(accuracies)])
            print(f"      📊 本轮假设准确率: {acc_str}")
            
            # 保存到文件
            save_path = os.path.join(game_log_dir, f"round_{round_num+1}_hypotheses.npy")
            np.save(save_path, hyp_stack)
            # ------------------------------------

            # D. Experiment / Test (Build a Koan)
            # 主动学习：找到 3 个假设分歧最大的未知公案
            # 所谓分歧最大，就是 mean(predictions) 最接近 0 (即 +1 和 -1 各半)
            
            # 1. 排除已知公案
            known_set = set(known_pos) | set(known_neg)
            unknown_mask = np.ones(self.atlas.num_koans, dtype=bool)
            unknown_mask[list(known_set)] = False
            unknown_indices = np.where(unknown_mask)[0]
            
            # 2. 计算分歧
            hyp_matrix = np.array(current_hypotheses) # (3, N)
            votes = np.sum(hyp_matrix, axis=0) # (N,) range [-3, 3]
            disagreement = -np.abs(votes) # 绝对值越小(越接近0)，分歧越大
            
            # 只在未知公案中找
            best_test_idx = unknown_indices[np.argmax(disagreement[unknown_indices])]
            
            # E. Ground Truth Check
            truth = gt_vector[best_test_idx]
            is_pos = (truth > 0)
            
            if is_pos:
                known_pos.append(best_test_idx)
                result_str = "SUCCESS (符合)"
            else:
                known_neg.append(best_test_idx)
                result_str = "FAIL (不符)"
                
            print(f"   🔄 Round {round_num+1}: 检验 Koan #{best_test_idx} -> {result_str} | 当前已知: {len(known_pos)}正/{len(known_neg)}反")

            # F. Slow Dynamics (Update Attention)
            # [Audit Fix] 显式选择能量最低的假设用于学习，而非简单的列表第一个
            # 能量越低，说明越符合当前的物理约束和观测数据
            best_h = min(current_hypotheses, key=lambda h: self.physics.compute_energy(h))
            
            current_attention = self.learner.update_attention(current_attention, best_h, self.physics)
            
            # print(f"      🧠 Attention Updated: Color={current_attention[0]:.2f}, Size={current_attention[1]:.2f}, ...")

        # --- Final Evolution (Round 8 Logic) ---
        print("   ⚡ 执行最终动力学演化...")
        
        # 1. Update Physics one last time
        dist_matrix = self.atlas.get_weighted_distance_matrix(current_attention)
        self.physics.update_couplings(dist_matrix)
        self.physics.set_pinning_field(known_pos, known_neg, round_num=7)
        
        # 2. Generate Final 3 Hypotheses
        chains = []
        mcmc_steps_override = self.atlas.num_koans * 20  # [Audit Fix] 保持一致的高步数
        
        for _ in range(Config.NUM_CHAINS):
            init_spins = np.random.choice([-1.0, 1.0], size=self.atlas.num_koans)
            final_spins = self.solver.run_glauber(self.physics, init_spins, steps=mcmc_steps_override)
            chains.append(final_spins)
            
        selected_indices, _ = self.dpp.select_subset(chains, self.physics)
        final_3_hypotheses = [chains[i] for i in selected_indices]
         # 补全
        while len(final_3_hypotheses) < 3:
            final_3_hypotheses.append(chains[np.random.randint(len(chains))])
            
        # --- [New] 记录最终轮次假设 ---
        final_stack = np.stack(final_3_hypotheses)
        final_accs = np.mean(final_stack == gt_vector, axis=1)
        acc_str = " | ".join([f"H{i}: {acc*100:.1f}%" for i, acc in enumerate(final_accs)])
        print(f"      📊 最终假设集准确率: {acc_str}")
        
        np.save(os.path.join(game_log_dir, "round_final_hypotheses.npy"), final_stack)
        # ----------------------------

        # 3. Merge (Voting)
        # 将3个假设合并为1个终极假设
        # Rule: Majority Vote. If sum > 0 then +1, else -1
        hyp_stack = np.stack(final_3_hypotheses) # (3, N)
        sum_votes = np.sum(hyp_stack, axis=0)
        ultimate_hypothesis = np.sign(sum_votes)
        # Fix sign(0) case (unlikely but possible) -> default to -1 or random
        ultimate_hypothesis[ultimate_hypothesis == 0] = -1.0
        
        # --- [New] 记录终极假设 ---
        np.save(os.path.join(game_log_dir, "ultimate_hypothesis.npy"), ultimate_hypothesis)
        # ------------------------

        # 4. Calculate Accuracy
        # Accuracy = (TP + TN) / Total
        matches = (ultimate_hypothesis == gt_vector)
        accuracy = np.mean(matches)
        
        print(f"   🎯 终极投票假设准确率: {accuracy*100:.2f}%")
        
        return accuracy, rule_name

if __name__ == "__main__":
    game = ZendoGame()
    game.play_all_games()