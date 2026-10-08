# CUTFGWD 两页数学版答辩 PPT 设计

## 沟通目标

两页内容面向导师与答辩评审。听众应在 2–3 分钟内理解：TGN 教师向图无关学生迁移三层知识；联合目标同时约束任务、概率、排序、关系结构与槽多样性；CUT-FGW 通过带时间先验的非平衡最优传输完成变长历史与固定关系槽的对齐。

## 输出约束

- 两页、16:9、深蓝科技风、中文标题。
- 公式与文字均使用可编辑文本，不使用公式截图。
- 不写推导和大段解释；公式是页面主体。
- 不展示实验结果，不增加项目代码中不存在的模块或结论。
- 公式字号不低于 16 pt，标题不低于 35 pt，并以投影可读性为优先。

## 第 1 页：框架与三层蒸馏信号

标题：**CUTFGWD：三层知识驱动的图无关时序蒸馏**

使用从左到右的流程：TGN 教师 → 三条训练期蒸馏通道 → LightST 学生。教师侧保留连续事件、TGN Memory、Temporal Neighbor 与 Graph Attention；学生侧保留 Node-Time Encoding、Residual MLP、Relation Slots 与候选打分。

三条通道直接显示紧凑公式：

\[
\mathcal L_{\mathrm{KD}}=T^2\,\mathrm{KL}(p^T\|p^S)
\]

\[
\mathcal L_{\mathrm{W1}}
=\sum_{k=1}^{C-1}w_k\left|\mathrm{CDF}^T_k-\mathrm{CDF}^S_k\right|
\]

\[
\mathcal L_{\mathrm{CUT-FGW}}
=\frac{\langle\Pi^\star,C_{\mathrm{FGW}}(\Pi^\star)\rangle}{\|\Pi^\star\|_1}
\]

底部仅保留三个短语：**概率对齐 / 排序对齐 / 关系结构对齐**。不再放解释性段落。

## 第 2 页：完整联合损失公式

标题：**联合蒸馏目标：任务监督 × 三层迁移 × 槽多样性**

顶部横贯整页展示联合目标：

\[
\mathcal L=\lambda_{\rm task}\mathcal L_{\rm CE}
+\lambda_{\rm logit}\mathcal L_{\rm KD}
+\lambda_{\rm rank}\mathcal L_{\rm W1}
+\lambda_{\rm rel}\mathcal L_{\rm CUT-FGW}
+\lambda_{\rm div}\mathcal L_{\rm div}
\]

左上区域显示 CE 与 Logit KD：

\[
\mathcal L_{\rm CE}=-\frac1B\sum_b\log
\frac{\exp(z^S_{b,y_b})}{\sum_c\exp(z^S_{b,c})}
\]

\[
p^T_{b,c}=\operatorname{softmax}(z^T_b/T)_c,\qquad
p^S_{b,c}=\operatorname{softmax}(z^S_b/T)_c
\]

\[
\mathcal L_{\rm KD}=\frac{T^2}{B}\sum_{b,c}p^T_{b,c}
\log\frac{p^T_{b,c}}{p^S_{b,c}}
\]

右上区域显示 Rank-Wasserstein：

\[
\pi_b=\operatorname{argsort}(z^T_b),\qquad
F^M_{b,k}=\sum_{r=1}^{k}p^M_{b,\pi_b(r)}
\]

\[
\mathcal L_{\rm W1}=\frac1B\sum_b\frac1{C-1}
\sum_{k=1}^{C-1}w_k\left|F^T_{b,k}-F^S_{b,k}\right|
\]

下半区域以 CUT-FGW 为主视觉：

\[
A_{ij}=(\tilde a^T_i-a^S_j)^2
\]

\[
G_{ij}(D^T,D^S,\Pi)=\sum_{k,l}
(D^T_{ik}-D^S_{jl})^2\Pi_{kl}
\]

\[
C_{\rm FGW}(\Pi)=\alpha A+(1-\alpha)
\frac{\lambda_mG(D^T_m,D^S_m,\Pi)
+\lambda_dG(D^T_d,D^S_d,\Pi)}{\lambda_m+\lambda_d}
\]

\[
K=P\odot\exp(-C_{\rm FGW}/\varepsilon),\qquad
\rho=\frac{\tau}{\tau+\varepsilon}
\]

\[
u\leftarrow(a/(Kv))^\rho,\qquad
v\leftarrow(b/(K^\top u))^\rho,\qquad
\Pi^\star=\operatorname{diag}(u)K\operatorname{diag}(v)
\]

页脚展示槽多样性：

\[
\mathcal L_{\rm div}=\frac1{BS(S-1)}
\sum_b\sum_{i\ne j}\cos^2(s_{b,i},s_{b,j})
\]

## 视觉层级

- 蓝色：任务监督与教师；天蓝色：Logit KD；紫色：Rank-Wasserstein；青绿色：CUT-FGW 与学生；粉色：多样性。
- 第 1 页以三条公式通道为视觉中心；第 2 页以联合目标与 CUT-FGW 公式组为视觉中心。
- 页面使用平面分区和细分隔线，避免重复卡片化和大段说明。

## 验收标准

- PPTX 恰好两页，且原文件被新的数学增强版替换。
- 所有上述公式均出现，且没有推导性段落。
- 公式、标题、教师/学生模块无裁切、越界或意外重叠。
- PowerPoint 中公式和标签可编辑。
- `slides_test.py` 不报告页面越界；两页渲染图逐页检查通过。
