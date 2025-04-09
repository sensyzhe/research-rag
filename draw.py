import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def extract_data_from_log(log_file):
    # 初始化数据列表
    similarity_thresholds = []
    context_recall = []
    faithfulness = []
    factual_correctness = []
    
    # 编译正则表达式模式
    pattern = r"相似度阈值 ([\d.]+) 的评估结果: {'context_recall': ([\d.]+), 'faithfulness': ([\d.]+), 'factual_correctness\(mode=f1\)': ([\d.]+)}"
    
    # 读取日志文件
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                threshold = float(match.group(1))
                cr = float(match.group(2))
                fh = float(match.group(3))
                fc = float(match.group(4))
                
                similarity_thresholds.append(threshold)
                context_recall.append(cr)
                faithfulness.append(fh)
                factual_correctness.append(fc)
    
    return similarity_thresholds, context_recall, faithfulness, factual_correctness

def plot_metrics(similarity_thresholds, context_recall, faithfulness, factual_correctness, output_prefix):
    # 创建折线图
    plt.figure(figsize=(15, 8))
    
    # 绘制三条线
    plt.plot(similarity_thresholds, context_recall, 'o-', label='Context Recall', linewidth=2)
    plt.plot(similarity_thresholds, faithfulness, 's-', label='Faithfulness', linewidth=2)
    plt.plot(similarity_thresholds, factual_correctness, '^-', label='Factual Correctness', linewidth=2)
    
    # 设置图表属性
    plt.title('RAG System Evaluation Metrics vs Similarity Threshold', fontsize=14)
    plt.xlabel('Similarity Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置y轴范围
    plt.ylim(-0.1, 1.0)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图表
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已生成：{output_prefix}.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python draw.py <日志文件名>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    try:
        # 从日志文件中提取数据
        similarity_thresholds, context_recall, faithfulness, factual_correctness = extract_data_from_log(log_file)
        
        # 生成输出文件名前缀（使用日志文件名，去掉.log后缀）
        output_prefix = log_file.rsplit('.', 1)[0]
        
        # 绘制图表
        plot_metrics(similarity_thresholds, context_recall, faithfulness, factual_correctness, output_prefix)
        
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1) 