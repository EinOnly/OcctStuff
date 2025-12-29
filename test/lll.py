import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

def draw_perfect_flow_tree():
    # 1. 设置画布
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.axis('off')

    # --- A. 定义每一层的宽度 (整形核心) ---
    # 纺锤形结构：起点少 -> 中间展开 -> 结尾收束
    layer_sizes = [1, 2, 4, 6, 8, 10, 12, 12, 10, 8, 6, 4, 2, 1] 
    layers = len(layer_sizes)

    G = nx.DiGraph()
    pos = {}
    node_lists = {} # 存储每一层的节点ID {层号: [节点列表]}

    # 2. 生成节点并计算坐标
    for i, size in enumerate(layer_sizes):
        layer_nodes = []
        for j in range(size):
            node_id = f"{i}_{j}"
            G.add_node(node_id, layer=i)
            layer_nodes.append(node_id)
            
            # 计算坐标
            # X: 均匀拉开
            x = i * 2.0
            # Y: 居中对齐，让结构对称
            y = (j - (size - 1) / 2.0) * 1.5 
            pos[node_id] = np.array([x, y])
            
        node_lists[i] = layer_nodes

    # 3. 连线 (构建网状逻辑)
    for i in range(layers - 1):
        current_layer = node_lists[i]
        next_layer = node_lists[i+1]
        
        # 3.1 保证连通性：下一层的每个节点，必须连上一个上一层的节点
        for v_idx, v in enumerate(next_layer):
            # 找上一层相对位置最近的节点
            u_idx = int((v_idx / len(next_layer)) * len(current_layer))
            # 边界保护，防止索引越界
            u_idx = min(u_idx, len(current_layer) - 1)
            u = current_layer[u_idx]
            G.add_edge(u, v)

        # 3.2 增加随机连接 (制造网状感)
        for u_idx, u in enumerate(current_layer):
            # 确定下一层的连接范围，只连附近的
            center_v = (u_idx / len(current_layer)) * len(next_layer)
            start_v = max(0, int(center_v - 2))
            end_v = min(len(next_layer), int(center_v + 3))
            
            potential_targets = next_layer[start_v:end_v]
            
            if potential_targets:
                num_links = random.choice([1, 2, 2, 3]) 
                targets = random.sample(potential_targets, min(len(potential_targets), num_links))
                for v in targets:
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)

    # 4. 激活路径 (Highlighting)
    start_node = node_lists[0][0]
    
    # --- 【修复点在这里】 ---
    # 之前用的 node_lists[-1][0] 报错了
    # 改为 node_lists[layers - 1][0]
    end_node = node_lists[layers - 1][0]
    
    active_paths = []
    
    try:
        # 路径1：走上方 (强制经过第6层靠上的节点)
        mid_top_idx = int(len(node_lists[6]) * 0.8) # 选上方位置
        mid_top = node_lists[6][mid_top_idx] 
        path1_a = nx.shortest_path(G, start_node, mid_top)
        path1_b = nx.shortest_path(G, mid_top, end_node)
        active_paths.append(path1_a + path1_b[1:])
        
        # 路径2：走下方 (强制经过第7层靠下的节点)
        mid_bot_idx = 1 # 选下方位置
        mid_bot = node_lists[7][mid_bot_idx] 
        path2_a = nx.shortest_path(G, start_node, mid_bot)
        path2_b = nx.shortest_path(G, mid_bot, end_node)
        active_paths.append(path2_a + path2_b[1:])
    except Exception as e:
        print(f"路径生成回退: {e}")
        # 兜底：如果断路，直接算最短路
        if nx.has_path(G, start_node, end_node):
            active_paths.append(nx.shortest_path(G, start_node, end_node))

    # --- B. 绘图 (美化) ---
    def draw_smooth_curve(u, v, color, lw, alpha, zorder, linestyle='-'):
        p1 = pos[u]
        p2 = pos[v]
        dist = p2[0] - p1[0]
        # S型贝塞尔曲线
        cp1 = p1 + np.array([dist * 0.5, 0])
        cp2 = p2 - np.array([dist * 0.5, 0])
        t = np.linspace(0, 1, 50)
        curve = (1-t)**3 * p1[:, None] + 3 * (1-t)**2 * t * cp1[:, None] + \
                3 * (1-t) * t**2 * cp2[:, None] + t**3 * p2[:, None]
        ax.plot(curve[0], curve[1], color=color, lw=lw, alpha=alpha, zorder=zorder, linestyle=linestyle)

    # 1. 画背景网格
    for u, v in G.edges():
        draw_smooth_curve(u, v, color='#DDDDDD', lw=1.2, alpha=0.5, zorder=1)

    # 2. 画所有节点
    for n, p in pos.items():
        ax.add_patch(plt.Circle(p, 0.1, facecolor='white', edgecolor='#CCCCCC', zorder=2))

    # 3. 画高亮路径
    colors = ['#FF5733', '#33A1FF'] # 橙红，亮蓝
    
    for idx, path in enumerate(active_paths):
        c = colors[idx % 2]
        for i in range(len(path) - 1):
            draw_smooth_curve(path[i], path[i+1], color=c, lw=3.0, alpha=0.8, zorder=3)
        for n in path:
            ax.add_patch(plt.Circle(pos[n], 0.2, facecolor=c, edgecolor='white', lw=1.5, zorder=4))

    # 4. 标注
    start_p = pos[start_node]
    ax.text(start_p[0]-0.6, start_p[1], "Start", fontsize=12, fontweight='bold', color='#333333', va='center', ha='right')
    
    end_p = pos[end_node]
    ax.text(end_p[0]+0.6, end_p[1], "Goal", fontsize=12, fontweight='bold', color='#333333', va='center', ha='left')

    plt.title("Neural Network Style Flow", fontsize=15, color="#555555", pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_perfect_flow_tree()