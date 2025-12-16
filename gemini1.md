Your project has achieved a stunning level of maturity and delivered multiple key breakthroughs in architecture search, particularly with the GQA/MQA scaling and the discovery of the Mamba quality advantage. 

The remaining open tasks are focused on extreme optimization and the final implementation needed to scale your research efficiently.

Here is a prioritized list of your next steps, focusing on open issues and immediate research needs:

---

## 🚀 Next Priorities: Optimization, Scale, and Search Strategy

### Priority 1: Optimization and Performance

Your fastest architectures (GQA/MQA) are now fully validated, but your highest-quality architectures (Mamba-based) are still bottlenecked on time.

| ID | Title | Domain | Next Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **arcfusion-8ys** | Implement Mamba speedups (parallel scan, CUDA kernels) | **Optimization / Kernel** | **Implement Mamba's core speedups.** This is the biggest single performance gain remaining. Reducing Mamba's training time (currently $706s$) will make your quality leaders (Pure Mamba, Mamba5to1) far more efficient for production and hybrid use. | **Crucial for ROI:** Directly improves the efficiency of your highest-quality models. |
| **arcfusion-ekl** | Ensure Claude compaction happens before training, not during | **Bug / Infrastructure** | **Resolve the Claude compaction bug.** This is a necessary stability fix to prevent wasting money on interrupted Modal GPU runs, which is especially critical now that you are running fast, deep GQA experiments. | **Cost Control & Stability:** Ensures your expensive A100 GPU time is not wasted. |

### Priority 2: Strategic Research & Experimentation

Your latest scaling results (`arcfusion-gzl`) found that MHA is *better* than GQA at depths of 24 layers and higher. This counter-intuitive finding requires follow-up research to exploit the discovered crossover point.

| ID | Title | Domain | Next Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **arcfusion-zdr** | Brainstorm: Patch Transformer weaknesses... | **Experimentation** | **Exploit the MHA/GQA Crossover Point:** Design hybrid experiments specifically around the **18-24 layer depth range** where MHA takes the lead. Test hybrid architectures (Mamba/MHA) at these high depths to see if the combination can push the quality curve even further than pure MHA. | **Data-Driven Search:** Leverages the findings from `arcfusion-gzl` to target a known area of architectural instability/opportunity. |
| **Open (New Task)** | **Re-run Mamba at High Depth (24L+)** | **Push Pure Mamba to extreme depths.** The MHA vs GQA comparison went to 32L, but pure Mamba's maximum depth was not confirmed. Test the limits of your quality leader (Mamba) at 24 and 32 layers to see if its quality advantage holds or if MHA scales better. | **Boundary Testing:** Defines the maximum potential of your best architecture before moving on to new component families. |

### Priority 3: Infrastructure and Housekeeping

These are essential tasks for preparing for the next phase of agent-driven development and cost management.

| ID | Title | Domain | Next Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **arcfusion-a8o** | Multi-agent autonomous development system | **Automation / Design** | **Begin implementation of the MVP Worker Agent** (Task 1 from the description). The architecture is defined; now implement the first agent that can claim an issue, run a Composer dream, and log the results. | **Next Phase of Project:** Starts the move toward autonomous architecture search and development. |
| **arcfusion-l3i** | Cache wiki training data for faster experiments | **Infrastructure** | **Implement full local caching of the WikiText-2 dataset** to eliminate all data download time from training runs. | **Micro-Optimization:** Eliminates the final few seconds of data loading from every single run, maximizing GPU efficiency. |
| **arcfusion-gb6** | Evaluate Modal vs other GPU services (Lambda, RunPod, etc) | **Cost Control** | **Conduct the cost evaluation.** Given the successful deployment of the A100, compare Modal's pricing model to alternatives for batch training runs. | **Fiduciary Duty:** Ensure compute budget is maximized for the coming wave of automated experiments. |
| **arcfusion-2si** | Improve paper ingestion workflow - API key handling | **DX / Setup** | **Implement the `.env` file support** for the Anthropic API key to streamline the paper ingestion research step. | **Improved Developer Experience:** Simplifies setup for new papers. |

***

### ✅ Summary of Immediate Next Step

The most impactful move right now is to **address the Mamba kernel bottleneck** while simultaneously launching the next set of targeted research experiments.

**Start with: `arcfusion-8ys` (Implement Mamba speedups).**