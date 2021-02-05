
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28­¿å@ß±H›«1bcluster_1_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28›Ù¼@…!H½Ý!bcluster_2_1/xla_runh
F
select_and_scatter_533*28Üª@Ýœ Hœé bcluster_1_1/xla_runh
9
	fusion_15*28Ûú@½ˆHºbcluster_1_1/xla_runh
9
	fusion_38*28£®ì@ŸHý”bcluster_2_1/xla_runh
9
	fusion_11*28¥åÓ@Þ—HÝÌbcluster_1_1/xla_runh
8
fusion_6*28æÒË@ýêH¾“bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Ë€”@Þ«HÝ„bcluster_1_1/xla_runh
\
sgemm_32x32x32_NN_vec*28¬Ã¹@ 3HàÕXb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28Úþ”@€(H ¡bCudnnRNNh‰
9
	fusion_25*28÷¼þ@¾ïHÞ¾bcluster_2_1/xla_runh
9
	fusion_24*28Ü¤é@¾ÄH¾Òbcluster_2_1/xla_runh
9
	fusion_20*28û¾ç@¾ÊH¿Øbcluster_2_1/xla_runh
9
	fusion_16*28ÂÓÉ@¾ºH¿óbcluster_1_1/xla_runh
A
reduce_window_119*28Ä÷¤@ßÜHß¢bcluster_2_1/xla_runh
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28§ìþ@¿¾
Hßébcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28‡Æþ@ŸÓHŸ³bcluster_2_1/xla_runh
Ä
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28è³ó@€H †bCudnnRNNhá
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ÇÃç@€H zXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_19*28È´Ý@¿’
Hßç
bcluster_2_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28«òÏ@ÿýHŸbcluster_1_1/xla_runh
9
	fusion_23*28íæÎ@ÿ¿	H¿³
bcluster_2_1/xla_runh
8
reduce_3*28ì•Æ@ÿ—	H¿é	bcluster_1_1/xla_runh
8
reduce_4*28íïÄ@€Š	H¿Œ
bcluster_1_1/xla_runh
8
fusion_1*28¬úÀ@¿äHÿÑ	bcluster_2_1/xla_runh
8
reduce_1*28ÎÉº@ ‹H¿‰
bcluster_2_1/xla_runh
8
reduce_5*28¹@ßÉHŸ	bcluster_1_1/xla_runh
6
reduce*28Í±²@ÿ‰HÿÊ	bcluster_2_1/xla_runh
9
	fusion_39*28Ï±˜@ŸòHŸËbcluster_2_1/xla_runh
9
	fusion_23*28––•@ íHÿžbcluster_1_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ö¨}@ ½H ®bcluster_2_1/xla_runh
8
	fusion_28*28ø¬{@ÀÖH€Šbcluster_1_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28‘—z@à-HŸ†Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28·q@ÿ™H Ìbcluster_1_1/xla_runh
8
	fusion_32*28º»b@ÀÀHàñbcluster_1_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28·œ\@ úHàùb2model/dropout/dropout/random_uniform/RandomUniformh
E
select_and_scatter_313*28´¯R@ßâHßbcluster_1_1/xla_runh
5
fusion*28¹ëQ@€ÒHŸ»bcluster_2_1/xla_runh
8
	fusion_12*28üÎE@ „HÀàbcluster_2_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28Ù³C@àGHÀøbcluster_1_1/xla_runh*
8
	fusion_41*28×»8@ßÐH Ýbcluster_1_1/xla_runh
8
	fusion_37*28™Ÿ8@àÌHÀåbcluster_1_1/xla_runh
8
	fusion_40*28š¨7@À«Hßíbcluster_2_1/xla_runh
8
	fusion_33*28Ý±1@à‘H¿Óbcluster_1_1/xla_runh
7
fusion_3*28Ýæ*@€ìHŸ©bcluster_2_1/xla_runh
8
	fusion_54*28ºü(@€âHß‘bcluster_1_1/xla_runh
8
	fusion_15*28ž³(@ áH€Ábcluster_2_1/xla_runh
E
select_and_scatter_143*28¿‘ @À·HàÏbcluster_1_1/xla_runh
8
	fusion_16*28Ÿµ@€ H€Übcluster_2_1/xla_runh
7
reduce_1*28œÎ@ §HàÃbcluster_1_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ßü@ £H ¼btranspose_0h
8
	fusion_17*28º­@Ÿ¢H ¸bcluster_2_1/xla_runh
7
reduce_3*28ÜŒ@ÀŠH€Ñbcluster_2_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Þï@€”H€Îb4model/dropout_1/dropout/random_uniform/RandomUniformh
8
	fusion_27*28ü÷@¿H€ºbcluster_2_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ßß@à’H€›b"gradients/transpose_grad/transposeh
6
fusion_4*28ÛÖ@¿}Hß—bcluster_3_1/xla_runh
?
reduce_window_193*28€Ã@àzH ›bcluster_2_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28¢@À{HßŠXb(gradients/CudnnRNN_grad/CudnnRNNBackproph
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28ý‚@à"HàzbCudnnRNNh*
4
reduce*28þå@ßoH Šbcluster_1_1/xla_runh
7
	fusion_11*28à®@àaHà¤bcluster_3_1/xla_runh
6
reduce_2*28ü@ mH ƒbcluster_2_1/xla_runh
7
	fusion_11*28Ûï@¿XHà¨bcluster_2_1/xla_runh
6
	fusion_56*28ž—@ÀiH€vbcluster_1_1/xla_runh
5
reduce_2*28¾È@€cHàhbcluster_1_1/xla_runh
5
fusion_2*28À•@ RH€zbcluster_2_1/xla_runh
5
fusion_5*28¾û@à^Hàfbcluster_2_1/xla_runh
7
	fusion_58*28þã@€OH ‚bcluster_1_1/xla_runh
5
fusion_6*28ÞÞ@À]Hßhbcluster_0_1/xla_runh
3
fusion*28¼…@àWH `bcluster_4_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ß„@àWH€hb4model/dropout_2/dropout/random_uniform/RandomUniformh
5
reduce_5*28Ç@¿GHÿqbcluster_2_1/xla_runh
5
reduce_4*28Þ¯@ÀSH€dbcluster_2_1/xla_runh
6
	fusion_24*28þÃ@€KHÀdbcluster_1_1/xla_runh
5
fusion_7*28Ÿ¸@€MHà^bcluster_1_1/xla_runh
>
reduce_window_263*28½¨@ :HÀXbcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ßü@ IH [b%Adam/Adam/update_12/ResourceApplyAdamh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Àé@€IH€Sb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_43*28ßá@àJH Rbcluster_1_1/xla_runh
6
	fusion_23*28 ¹@€GH€Ubcluster_3_1/xla_runh
5
fusion_9*28€é@€@H Obcluster_2_1/xla_runh
3
fusion*28¿Ó@ÀCH€Kbcluster_6_1/xla_runh
6
	fusion_60*28þÁ@¿CHÀNbcluster_1_1/xla_runh
3
fusion*28Þš@à>HÿMbcluster_5_1/xla_runh
6
	fusion_10*28¿@à1H€qbcluster_2_1/xla_runh
9
fusion_33__2*28ŸÁ
@ 0H€Sbcluster_3_1/xla_runh
5
fusion_8*28Ÿ­
@ ;HÀVbcluster_2_1/xla_runh
6
	fusion_26*28¿Š
@ 5Hàsbcluster_2_1/xla_runh
6
	fusion_18*28ßú	@€1Hàkbcluster_2_1/xla_runh
4
copy_57*28¿½	@À.H >bcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾±	@À,H€?b%Adam/Adam/update_14/ResourceApplyAdamh
4
copy_50*28Þ¤	@à*HÀYbcluster_2_1/xla_runh
6
	fusion_42*28¾æ@à HŸgbcluster_2_1/xla_runh
6
	fusion_61*28À×@à2H€8bcluster_1_1/xla_runh
4
add_297*28¿È@Ÿ+H <bcluster_3_1/xla_runh
6
	fusion_30*28À¾@ /H Jbcluster_3_1/xla_runh
6
	fusion_16*28¿¾@À1H 7bcluster_3_1/xla_runh
4
copy_57*28Ÿ½@À+Hà@bcluster_2_1/xla_runh
6
	fusion_41*28ÿ•@à H€6bcluster_2_1/xla_runh
3
fusion*28À‰@à&Hà@bcluster_3_1/xla_runh
6
	fusion_43*28Àü@À#Hà8bcluster_2_1/xla_runh
6
	fusion_45*28¿â@€&H 8bcluster_2_1/xla_runh
4
copy_72*28ÿ¯@À)Hà?bcluster_1_1/xla_runh
6
	fusion_10*28Ÿ†@à"HÀ7bcluster_3_1/xla_runh
6
	fusion_12*28€à@À'H ,bcluster_0_1/xla_runh
6
	fusion_33*28àÖ@à!H€1bcluster_3_1/xla_runh
9
fusion_33__1*28½½@À!H€Cbcluster_3_1/xla_runh
6
	fusion_11*28à£@ %HÀ(bcluster_0_1/xla_runh
6
	fusion_36*28Ÿ€@€ H€1bcluster_3_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28¿þ@€"Hà(bcluster_2_1/xla_runh
6
	fusion_13*28€þ@€"HÀ.bcluster_0_1/xla_runh
6
	fusion_50*28Àõ@À H (bcluster_1_1/xla_runh
5
fusion_1*28Àô@€ H (bcluster_3_1/xla_runh
3
add_38*28 ë@€"HÀ&bcluster_0_1/xla_runh
4
add_350*28€ß@à H€.bcluster_3_1/xla_runh
6
	fusion_21*28¾Ø@  HÀ*bcluster_3_1/xla_runh
3
add_70*28þÂ@à Hà#bcluster_0_1/xla_runh
6
	fusion_49*28€¾@ H -bcluster_3_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à‘@ Hà"bAssignAddVariableOp_7h
6
	fusion_47*28Àÿ@ÀHà$bcluster_3_1/xla_runh
4
add_338*28ÿü@àHÀ&bcluster_3_1/xla_runh
4
add_375*28Àø@€HÀ!bcluster_3_1/xla_runh
6
	fusion_44*28ßõ@ŸH€!bcluster_2_1/xla_runh
4
add_363*28Àõ@àHà$bcluster_3_1/xla_runh
6
	fusion_42*28Þó@ Hà$bcluster_3_1/xla_runh
4
slice_1*28ßÕ@ÀHàbcluster_5_1/xla_runh
6
	fusion_27*28€š@€Hàbcluster_3_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 …@€H€bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 #@ #H #b
LogicalAndh