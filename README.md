# Longitudinal Brain Segmentation with Temporal Consistency for Neurodegenerative Analysis
In this work, we introduce two novel temporal consistency loss functions for longitudinal brain segmentation: a volume-based loss (VL) that promotes consistent volume trajectories without population-level normalization and a signed distance field (SDF) loss that enforces expected structural shrinkage without relying on age or time intervals. Unlike prior methods, our framework handles variable gaps between timepoints, making it more applicable to real-world clinical data. The paper was accepted into the Learning with Longitudinal Medical Images and Data MICCAI 2025 Workshop.

# The Codes

###  CFSegNet 
The folder contains my implementation of the CFSegNet (Wei et al., 2021). It is important to note that we provided the reimplementation of the CFSegNet model, as the original paper did not provide any code. We changed the baseline model. The is the modified version where the SC loss has been adapted to handle multiple time points.

```bibtex
@inproceedings{CFSegNet,
  author    = {Wei, J. and Shi, F. and Cui, Z. and Pan, Y. and Xia, Y. and Shen, D.},
  title     = {Consistent Segmentation of Longitudinal Brain MR Images with Spatio-Temporal Constrained Networks},
  booktitle = {MICCAI 2021},
  pages     = {89--98},
  publisher = {Springer},
  address   = {Cham},
  year      = {2021}
}
```

###  Ours 
The folder contains our implementation of the proposed modle with the new intorduced VL and SDF loss.

> The citation for this work will be made available when it is published after the MICCAI conference.


