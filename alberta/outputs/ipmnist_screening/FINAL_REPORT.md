# Beat-SOTA campaign: final report

Baseline: published-config UPGD-W, our 10-seed exact reproduction = 0.7791
(published ICLR-2024 figure ~0.78). All results development-grade/nonpromoting.

## Full-protocol (200-task) confirmations

| arm | n | mean | stderr | vs SOTA | verdict |
|---|---|---|---|---|---|
| upgd_ema_norm | 6 | 0.85359 | 0.00010 | +0.0745 | BEATS-SOTA |
| adamw_cbp | 10 | 0.79876 | 0.00009 | +0.0197 | BEATS-SOTA |
| upgd_w_wd0005 | 10 | 0.78431 | 0.00014 | +0.0052 | BEATS-SOTA |
| upgd_l2init | 3 | 0.78042 | 0.00030 | +0.0013 | BEATS-SOTA |
| upgd_idbd | 3 | 0.77895 | 0.00020 | -0.0002 | TIES |

Pool64 paired controls (seeds 0-2): {1: 0.7787729776000001, 2: 0.77910898075, 0: 0.7786109820999999}; exact partials 0.77906/0.77903/0.77932;
pool-vs-exact delta -0.00012. adamw_cbp + idbd arms ran exact/step mode.

## Screening ranked table (60-task validated proxy, paired vs upgd_w_control)

| arm | mean | paired | candidate |
|---|---|---|---|
| upgd_ema_norm | 0.8529 | +0.0751 | True |
| upgd_ema_norm_sigma0 | 0.8520 | +0.0742 | True |
| upgd_ema_norm_wd0005 | 0.8469 | +0.0691 | True |
| upgd_ema_norm_lr0003 | 0.8421 | +0.0644 | True |
| adamw_cbp_ema_norm | 0.7995 | +0.0217 | True |
| adamw_cbp_r3e4 | 0.7983 | +0.0205 | True |
| adamw_cbp | 0.7965 | +0.0188 | True |
| adamw_cbp_m50 | 0.7962 | +0.0185 | True |
| adamw_cbp_m200 | 0.7959 | +0.0181 | True |
| adamw_cbp_noreset | 0.7955 | +0.0177 | True |
| adamw_cbp_r3e5 | 0.7905 | +0.0127 | True |
| upgd_w_wd0005 | 0.7833 | +0.0056 | True |
| upgd_l2init | 0.7792 | +0.0014 | False |
| upgd_idbd | 0.7778 | +0.0001 | False |
| upgd_w_control | 0.7778 | +0.0000 | False |
| upgd_alpha_utility | 0.7777 | -0.0000 | False |
| upgd_w_udecay099999 | 0.7772 | -0.0006 | False |
| upgd_w_localgate | 0.7770 | -0.0008 | False |
| upgd_idbd_meta1e2 | 0.7768 | -0.0010 | False |
| upgd_cbp | 0.7766 | -0.0012 | False |
| upgd_w_udecay0999 | 0.7764 | -0.0014 | False |
| upgd_w_sigma005 | 0.7760 | -0.0018 | False |
| upgd_ema_norm_lr003 | 0.7747 | -0.0030 | False |
| guarded_cbp_adam | 0.7723 | -0.0055 | False |
| upgd_w_wclip_k2 | 0.7721 | -0.0056 | False |
| upgd_w_wclip_k2_wd0 | 0.7709 | -0.0068 | False |
| upgd_w_sigma02 | 0.7689 | -0.0089 | False |
| upgd_w_wd002 | 0.7600 | -0.0178 | False |
| upgd_w_fade_head | 0.7587 | -0.0190 | False |
| adamw_control | 0.7556 | -0.0222 | False |
| upgd_w_wclip_k1_wd0 | 0.7453 | -0.0324 | False |
| upgd_w_sigma0 | 0.7429 | -0.0349 | False |
| upgd_w_wclip_k1 | 0.7396 | -0.0381 | False |
| upgd_autostep | 0.6863 | -0.0915 | False |
| upgd_w_idbd_swift | 0.6029 | -0.1749 | False |

## Mechanistic findings (see CONTINUAL_LEARNING_THEORY.md)

1. adamw_cbp 0.79876±0.00009 (n=10, exact, protocol-pure): CBP recycling fully
   arrests Adam plasticity decay; never previously run on this protocol.
2. upgd_ema_norm 0.85357 (n=3, protocol-extended): EMA input normalization
   transforms gated-SGD (+0.075); Alberta Step-1 tenet the published setup omitted.
3. Composition adamw_cbp_ema_norm 0.7995 ~ adamw_cbp: Adam second moment IS input
   conditioning; normalization redundant on Adam. Normalized gated-SGD > conditioned Adam+CBP.
4. upgd_ema_norm_sigma0 0.8520 ~ base: perturbation not load-bearing under conditioning
   (7x cheaper); 200-task confirmation + gate ablation (sgd_ema_norm) in flight.
5. Refutation: guarded_cbp_adam -0.0055 — protection costs on no-recurrence protocols.
6. Honest negatives: weight clipping hurts on relu; UPGD+CBP redundant; FADE-head,
   SwiftTD-IDBD ports negative; Autostep batch-1 meta-instability.