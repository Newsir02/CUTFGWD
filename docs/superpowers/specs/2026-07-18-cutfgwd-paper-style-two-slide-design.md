# CUTFGWD 论文风格两页 PPT 设计

## 沟通目标

面向导师与答辩评审，用两页内容说明 CUTFGWD 的教师—学生蒸馏框架与核心损失。公式必须像论文中一样简洁、正式：每个关键损失只保留一条核心定义，最后用一条总损失汇总，不展示推导和中间求解步骤。

## 唯一视觉参考

- 来源：`D:\文件\NormalFile\周汇报\path.pptx`。
- 第 1 页继承参考文件第 2 页“模型架构”的页面骨架。
- 第 2 页继承参考文件第 3 页“Dual-Grained Structural Distillation”的页面骨架。
- 保留参考文件的白色背景、西南大学标识、左上标题箭头、细灰分隔线、蓝色页脚和黑色论文公式排版。
- 不混入原深蓝科技风，也不使用其他模板。
- 原始 `path.pptx` 保持不变，输出独立文件 `artifacts/cutfgwd_two_slide_paper_style.pptx`。

## 第 1 页：模型架构

标题：**模型架构**

画面结构沿用参考文件第 2 页的左右流程：

1. 左侧：TGN Teacher，输入连续时间交互事件，包含 TGN Memory、Temporal Neighbor 和 Graph Attention。
2. 中间：三条教师到学生的知识迁移路径，只标记损失名称：
   - `L_KD`：概率分布蒸馏；
   - `L_W1`：候选排序蒸馏；
   - `L_CUT-FGW`：历史关系结构蒸馏。
3. 右侧：Graph-free Student，包含 Node-Time Encoding、Residual MLP、Relation Slots 和候选打分。

本页不展示展开公式，仅用三个损失符号说明迁移通道。

## 第 2 页：蒸馏目标

标题：**CUTFGWD 蒸馏目标**

页面采用参考文件第 3 页的论文公式排版：白底、黑色数学公式、较大的留白、每条公式独立成组。只展示以下四组公式。

### Logit KD

\[
\mathcal L_{\mathrm{KD}}
=T^2D_{\mathrm{KL}}(p^T\|p^S)
\]

### Rank-Wasserstein

\[
\mathcal L_{\mathrm{W1}}
=\sum_{k=1}^{C-1}w_k\left|F_k^T-F_k^S\right|
\]

### CUT-FGW

\[
\mathcal L_{\mathrm{CUT-FGW}}
=\frac{\langle\Pi^\star,C_{\mathrm{FGW}}(\Pi^\star)\rangle}
{\|\Pi^\star\|_1}
\]

### Total Loss

\[
\mathcal L_{\mathrm{Total}}
=\mathcal L_{\mathrm{Task}}
+\lambda_1\mathcal L_{\mathrm{KD}}
+\lambda_2\mathcal L_{\mathrm{W1}}
+\lambda_3\mathcal L_{\mathrm{CUT-FGW}}
+\lambda_4\mathcal L_{\mathrm{div}}
\]

## 明确删除的内容

- 交叉熵展开式；
- 教师和学生概率的 softmax 定义；
- Rank-Wasserstein 的排序支撑和 CDF 展开定义；
- CUT-FGW 的属性代价、GW 结构代价和融合代价展开式；
- Unbalanced Sinkhorn 的核矩阵、缩放变量和迭代更新式；
- 槽多样性损失的展开式；
- 所有公式推导和解释性段落。

## 公式与文字格式

- 公式使用与参考文件相同的黑色论文式数学排版，变量为斜体，损失符号和上下标清晰。
- 公式应作为标准排版对象或高分辨率透明公式图置入继承的公式区域，不使用带花括号的普通文本伪公式。
- 页面可见正文以简短标题和模块标签为主，不使用大段文字。
- 不缩小到投影不可读的字号；如内容不适配，优先缩短标签和扩大留白。

## 验收标准

- 输出恰好两页。
- 两页均保留参考文件的校徽、校名、标题箭头和页脚风格。
- 第 1 页只有框架和三个损失名称，没有展开公式。
- 第 2 页只出现四组确认公式，没有中间推导式。
- 公式排版不存在普通文本下标花括号、裁切、重叠或越界。
- 独立渲染两页并逐页检查；`slides_test.py` 不报告越界。
