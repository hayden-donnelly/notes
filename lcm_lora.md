# LCM-LoRA: A Universal Stable-Diffusion Acceleration Module
[Paper](https://arxiv.org/abs/2311.05556)

## Abstract
- Latent Consistency Models (LCMs) are distilled from latent diffusion models and produce high quality
images in very few inference steps.
- The authors use LoRA to distill latent diffusion models into LCMs, thus reducing memory consumption.
- The LoRA parameters obtained through LCM distillation can be used to accelerate various stable 
diffusion models without training.
- The authors name this method LCM-LoRA.
- In contrast to numerical PF-ODE solvers (i.e. DDIM and DPM-Solver), LCM-LoRA can be viewed as a plug in
neural PF-ODE solver with strong generalization capabilities.

## Introduction
- Latent Diffusion Models (LDMs) can generate very high quality images, but their sampling process is
iterative and slow.
- One category of methods used to accelerate LDMs involves the use of advanced ODE-Solvers like DDIM, 
DPM-Solver, and DPM-Solver++.
- ODE-Solvers reduce the number of sampling steps required but are still quite computationally expensive.
- LCMs (which are inspired by consistency models) are another way to reduce the number of sampling steps
required.
- LCMs treat the reverse diffusion process as an augmented probability flow ODE (PF-ODE) problem.
- They predict the solution in the latent space, thus bypassing the need for iterative solutions through
numerical ODE-Solvers.
- The result is high quality image generation in only 1 to 4 inference steps.
- Latent Consistency Finetuning (LCF) has been developed as a method to fine-tune pretrained LCMs without
starting from the teacher diffusion model.
- This requirement for additional training presents a barrier to fast deployment of downstream LCMs.
- To solve this, the authors present LCM-LoRA, *a universal training-free acceleration module* that can be
plugged directly into various Stable-Diffusion (SD) fine-tuned models or SD LoRAs for fast inference with
minimal steps.

## Related Work
### Consistency Models
- Consistency models are a novel class of generative models that enhance sampling efficiency without
sacrificing the output quality.
- These models employ a consistency mapping technique that maps points along the ODE trajectory to their
origins.
- This enables one-step generation.
- LCMs have been developed within the text-to-image domain.
- By viewing the guided reverse diffusion process as the resolution of a PF-ODE, LCMs can predict the
solution of such ODEs in latent space.

### Parameter-Efficient Fine-Tuning
- Parameter Efficient Fine-Tuning (PEFT) enables model finetuning while limiting the number of parameters
required during training.
- This reduces computational load and storage demands.
- Low-Rank Adaptation (LoRA) is one technique under the PEFT umbrella.
- LoRA involves training a minimal set of parameters via the usage of low rank matrices which succintly
represent the required weight adjustments.
- In practice this means that only the low rank matrices are trained and most of the pre-trained weights
are left unchanged.

## LCM-LoRA
- Latency consistency distillation can be considered a fine-tuning process for LDMs since the distillation
process is carried out on top of the pretrained LDM.
- This allows for the usage of parameter efficient fine-tuning methods such as LoRA during the distillation
process.
- By incorporating LoRA into the distillation process, the quantity of trainable parameters is significantly
reduced, thereby also reducing the memory requirements of training and enabling the distillation of larger
models.
- LCM-LoRA parameters can be combined with other LoRA parameters which have been fine-tuned on datasets of 
particular styles in order accelerate sampling in those styles without additional training.
- Denoting the LCM-LoRA fine-tuned parameters as $\tau_{\text{LCM}}$ (the "acceleration vector"), and 
the LoRA parameters fine-tuned on a customized dataset as $\tau^{\prime}$ (the "style vector"), an LCM 
which generates customized images can be obtained as:
$`\theta^{\prime}_{\text{LCM}} = \theta_{\text{pre}} + \tau^{\prime}_{\text{LCM}}`$.
- $`\tau^{\prime}_{\text{LCM}} = \lambda_1\tau^{\prime} + \lambda_2\tau_{\text{LCM}}`$ is a linear
combination of the acceleration vector $`\tau_{\text{LCM}}`$ and style vector $`\tau^{\prime}`$.
- $`\lambda_1`$ and $`\lambda_2`$ are hyperparameters.
