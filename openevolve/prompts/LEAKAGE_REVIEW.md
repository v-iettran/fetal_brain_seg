# System-message leakage review

The file `system_message.txt` must be checked by a person who has **not** read
`PREREGISTRATION.md` § target list (or OpenEvolve_Spec_FeTA.md §3.1) before any
paid LLM call (zero-shot control, dry run, or main evolution).

Automated lint (`python -m openevolve.prompts.lint_system_message`) catches
direct target names. It cannot catch clever paraphrases. That is this review.

Checklist:

- [ ] The message does not name instance / group normalisation.
- [ ] The message does not name leaky ReLU, ELU, or GELU as the thing to use.
- [ ] The message does not mention strided convolution as a pooling replacement.
- [ ] The message does not mention deep / auxiliary supervision.
- [ ] The message does not mention Dice loss (or soft Dice).
- [ ] The message does not mention SGD, Nesterov, or polynomial LR decay.
- [ ] The message does not mention foreground oversampling / forced-foreground patches.
- [ ] The message does not mention elastic deformation, gamma, or low-res simulation
      as the augmentation to add.

After review, set the following line to YES and fill the reviewer field:

SIGNED_OFF=NO
REVIEWER=
DATE=
NOTES=
