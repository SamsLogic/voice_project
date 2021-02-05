

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28´™¶@½úHù Ebcluster_0_1/xla_runh*
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ƒÙ¼@¾Hüå0bcluster_0_1/xla_runh*
F
select_and_scatter_691*28¹§ï
@ºäAHÙÓBbcluster_0_1/xla_runh
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28ß¤Ï
@ýÃHü‹(bcluster_1_1/xla_runh*
9
	fusion_20*28¨ÏÅ	@Ûê9Hšð;bcluster_0_1/xla_runh
9
	fusion_24*28ˆÚ¼	@š¨9Hú›:bcluster_0_1/xla_runh
9
	fusion_15*28«¨¶	@û 9HšÐ9bcluster_0_1/xla_runh
9
	fusion_56*28ë€¶	@ûý8H›‚:bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Úãˆ@»ä/H›´2bcluster_0_1/xla_runh
9
	fusion_25*28‡¿ñ@ü)HüÆ+bcluster_0_1/xla_runh
9
	fusion_31*28Òú‘@ü%HüÞ%bcluster_1_1/xla_runh
\
sgemm_32x32x32_NN_vec*28’Ñˆ@À2H¿ºXb(gradients/CudnnRNN_grad/CudnnRNNBackprophê

9
	fusion_26*28¶Øú@üö#H¼é$bcluster_1_1/xla_runh
9
	fusion_30*28³ƒø@¼ï#Hý¼$bcluster_1_1/xla_runh
:
sgemm_32x32x32_NN_vec*28“›Ù@ÿ(H€·bCudnnRNNhê

£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28„„ä@ËH¾åbcluster_0_1/xla_runh
A
reduce_window_180*28èüÍ@ÝÕHž°bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28ÆË¼@Þ¹H½bcluster_1_1/xla_runh
9
	fusion_35*28Ž°ü@¾“H¾šbcluster_0_1/xla_runh
²
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28¼®Ý@àH «Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophÄ
9
	fusion_25*28¶á­@þãHþîbcluster_1_1/xla_runh
Ä
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28Õ„ª@àH€ŽbCudnnRNNhÄ
7
copy_66*28»‹@žÝH¾¶bcluster_1_1/xla_runh
9
	reduce_15*28»êŒ@žÔHžŸbcluster_0_1/xla_runh
8
fusion_1*28»æŠ@Þ¾Hþ“bcluster_1_1/xla_runh
8
reduce_2*28œÞ€@žïHžƒbcluster_1_1/xla_runh
8
reduce_3*28û¸€@þïHþýbcluster_1_1/xla_runh
9
	reduce_16*28›Ùÿ@žöH¾Òbcluster_0_1/xla_runh
9
	reduce_14*28›íý@žâHžñbcluster_0_1/xla_runh
8
fusion_8*28úý@¾üHß¨bcluster_0_1/xla_runh
9
	fusion_33*28ÝÀú@ÞÍHÞ¶bcluster_1_1/xla_runh
9
	fusion_22*28Ý€é@þëH¿Ïbcluster_1_1/xla_runh
9
	fusion_24*28ÞÍç@žÛHÞÂbcluster_1_1/xla_runh
w
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28úãå@ÀwHÿÉbcluster_0_1/xla_runh?
9
	fusion_51*28Áåª@žáHžäbcluster_0_1/xla_runh
9
	fusion_47*28ä»@¿´HŸëbcluster_0_1/xla_runh
F
select_and_scatter_442*28§œ–@ÿðHßÏbcluster_0_1/xla_runh
9
	fusion_42*28åŽ@ßÜHß‡bcluster_0_1/xla_runh
9
	fusion_57*28ãŽ‰@þ¨Hßòbcluster_1_1/xla_runh
9
	fusion_29*28…©ö@ÿ¬H¾bcluster_1_1/xla_runh
6
fusion*28æëò@ŸªHÿébcluster_1_1/xla_runh
[
sgemm_32x32x32_TN_vec*28‹äà@ÿ-HßÿXb(gradients/CudnnRNN_grad/CudnnRNNBackproph~
è
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ëÛÑ@ÿß	H¿¡
b6model_2/dropout_6/dropout/random_uniform/RandomUniformh
9
	fusion_38*28­ŠÍ@¿Ö	H¿î	bcluster_1_1/xla_runh
9
	fusion_36*28­ÒÌ@ßÎ	H¿ù	bcluster_1_1/xla_runh
9
	fusion_52*28íßÉ@Ÿš	Hß’
bcluster_0_1/xla_runh
9
	fusion_15*28«ÁÆ@¿˜	H¿ñ	bcluster_1_1/xla_runh
8
reduce_5*28îæ»@ß¿Hÿü	bcluster_1_1/xla_runh
8
reduce_4*28Ë—»@ÿÒHß¤	bcluster_1_1/xla_runh
9
	fusion_66*28îÏ·@ŸµH¿™	bcluster_0_1/xla_runh
9
	fusion_76*28ÐÅ«@ÿ‚H ½bcluster_0_1/xla_runh
9
	fusion_20*28°µ©@ÿÕHßÏbcluster_1_1/xla_runh
9
	fusion_70*28ò‹”@ŸìHÿ¯bcluster_0_1/xla_runh
9
	fusion_62*28ñÊ‘@¿HßÂbcluster_0_1/xla_runh
9
	fusion_19*28Ò“„@ÀöH Ãbcluster_1_1/xla_runh
7
fusion_3*28ºöe@Ÿ©HÀ¸bcluster_1_1/xla_runh
6
copy_58*28ô÷a@Ÿ™H¿ÿbcluster_1_1/xla_runh
@
reduce_window_303*28×‚`@ß¤HŸýbcluster_1_1/xla_runh
7
fusion_5*28ºì[@ßóH Übcluster_1_1/xla_runh
8
	reduce_11*28ºÚR@ÀÚHÀœbcluster_0_1/xla_runh
8
	fusion_85*28ô€N@ÿÄHà•bcluster_0_1/xla_runh
8
	fusion_58*28—äM@€¼HÀúbcluster_1_1/xla_runh
7
reduce_7*28›¾D@ÿêHŸÿbcluster_1_1/xla_runh
7
reduce_6*28šÌB@àåHÿÁbcluster_1_1/xla_runh
8
	fusion_18*28¼­B@ Hß¥bcluster_1_1/xla_runh
8
	reduce_10*28ýýA@àòH Ábcluster_0_1/xla_runh
8
	reduce_12*28ºA@àìHà§bcluster_0_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾ˆA@€}H€šbtranspose_0h*
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28÷Ú=@à†HÀ÷b"gradients/transpose_grad/transposeh*
E
select_and_scatter_193*28šª;@ÀÖH Œbcluster_0_1/xla_runh
ç
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ºò6@€šHÀìb6model_2/dropout_7/dropout/random_uniform/RandomUniformh
7
fusion_2*28úË6@€šHà¤bcluster_1_1/xla_runh
8
	fusion_39*28™Ò3@ÿ¥Hßñbcluster_1_1/xla_runh
8
	fusion_14*28›À1@àïHàÙbcluster_1_1/xla_runh
8
	fusion_83*28¼²/@à‹Hà×bcluster_0_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28ßˆ)@€"H ibCudnnRNNhT
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28¾‡)@ÀtHà‡Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph*
7
fusion_9*28›Ü'@àÁH€bcluster_1_1/xla_runh
6
copy_57*28Ý÷#@€ÎH¿íbcluster_1_1/xla_runh
6
copy_64*28½‰#@à¾HŸŠbcluster_0_1/xla_runh
8
	fusion_10*28¿Õ!@à¿HÀbcluster_1_1/xla_runh
8
	fusion_13*28¾ÿ@ßH ïbcluster_1_1/xla_runh
8
	fusion_11*28ùé@ÿ§HŸÝbcluster_1_1/xla_runh
6
reduce_9*28ÿÿ@àH€²bcluster_1_1/xla_runh
8
	fusion_88*28½±@€HŸ·bcluster_0_1/xla_runh
?
reduce_window_422*28Ÿ•@€ZH€Íbcluster_1_1/xla_runh
7
	reduce_11*28¿€@ pHÀ”bcluster_1_1/xla_runh
6
fusion_4*28ÛÆ@ÀxHÀ”bcluster_2_1/xla_runh
æ
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28þº@ yHà›b6model_2/dropout_8/dropout/random_uniform/RandomUniformh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ý÷@€ƒH€‰btranspose_9h
´
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Þó@€}HÀ‡b$gradients/transpose_9_grad/transposeh
7
	reduce_10*28¾È@ tHà—bcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ž‹@ nH€¨b%Adam/Adam/update_18/ResourceApplyAdamh
6
reduce_8*28»‘@¿lHÀ‚bcluster_1_1/xla_runh
6
fusion_8*28Û‹@ŸUHàbcluster_1_1/xla_runh
6
fusion_8*28¿µ@ ]HÀ›bcluster_2_1/xla_runh
3
fusion*28Ÿð@ \Hàkbcluster_8_1/xla_runh
4
fusion*28¿Í@ XHÀŽbcluster_9_1/xla_runh
¬
ƒ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28žæ@¿\H ebgradients/AddNh
3
fusion*28àÃ@€ZH€ibcluster_3_1/xla_runh
7
	fusion_53*28¿‹@ MHß‰bcluster_0_1/xla_runh
6
	fusion_16*28ÿ³@ÀLH€dbcluster_0_1/xla_runh
6
	fusion_43*28Ÿˆ@àLH nbcluster_0_1/xla_runh
4
copy_81*28ž„@¿KHàfbcluster_0_1/xla_runh
6
	fusion_21*28àý@€MH€lbcluster_1_1/xla_runh
6
	fusion_26*28àÐ@ LH ebcluster_0_1/xla_runh
6
	fusion_72*28€±@€OH Wbcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28€–@ KH€^b%Adam/Adam/update_19/ResourceApplyAdamh
4
copy_64*28üÊ@€AH ebcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28žÂ@àFH Qb%Adam/Adam/update_21/ResourceApplyAdamh
6
	fusion_64*28þì@€CHÀsbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 ä@ BHàLb%Adam/Adam/update_22/ResourceApplyAdamh
6
	fusion_59*28 ®@ #H€Kbcluster_1_1/xla_runh
6
	fusion_23*28¼«@àAHÿJbcluster_2_1/xla_runh
6
fusion_1*28þ£@€9HßNbcluster_12_1/xla_runh
5
fusion_5*28€€@€3HàLbcluster_0_1/xla_runh
6
	fusion_91*28Àî
@À>H€Dbcluster_0_1/xla_runh
4
fusion*28ŸÞ
@À?H€Hbcluster_11_1/xla_runh
4
fusion*28Þ½
@€<HàIbcluster_10_1/xla_runh
9
fusion_32__2*28¾¹
@€<H Hbcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Àó	@À8HàBb%Adam/Adam/update_23/ResourceApplyAdamh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 ‰	@ %H€@b%Adam/Adam/update_20/ResourceApplyAdamh
4
fusion*28¿È@ -HàGbcluster_12_1/xla_runh
6
	fusion_16*28€±@À/H 6bcluster_2_1/xla_runh
4
add_258*28 ¬@À+HÀCbcluster_2_1/xla_runh
6
	fusion_92*28€¦@à*HÀEbcluster_0_1/xla_runh
6
	fusion_29*28¿ƒ@ -H€5bcluster_2_1/xla_runh
6
	fusion_32*28ÿ½@€%HÀObcluster_1_1/xla_runh
3
fusion*28€»@€$H€;bcluster_2_1/xla_runh
6
	fusion_62*28 «@ 'HÀ6bcluster_1_1/xla_runh
6
	fusion_60*28Þ@ H Lbcluster_1_1/xla_runh
6
	fusion_65*28Àœ@€'Hà2bcluster_1_1/xla_runh
6
	fusion_13*28€†@À(H€/bcluster_4_1/xla_runh
6
	fusion_61*28€ƒ@ÀH€5bcluster_1_1/xla_runh
5
fusion_6*28à‚@À#HÀ7bcluster_2_1/xla_runh
6
	fusion_47*28€Ê@À#H€@bcluster_2_1/xla_runh
6
	fusion_63*28ÿÀ@€ Hà>bcluster_1_1/xla_runh
6
	fusion_32*28Ÿ½@À!H 1bcluster_2_1/xla_runh
4
copy_50*28ÿ±@Ÿ$HÀ-bcluster_1_1/xla_runh
4
copy_98*28¿§@À"Hà-bcluster_0_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¿ž@ #HÀ-b
div_no_nanh
6
	fusion_34*28žœ@àH¿-bcluster_1_1/xla_runh
6
	fusion_12*28€’@€ Hà-bcluster_4_1/xla_runh
6
	fusion_21*28¿Œ@Ÿ H€*bcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28€Š@À#H€*bcluster_1_1/xla_runh
6
	fusion_48*28€þ@à!Hà-bcluster_2_1/xla_runh
6
	fusion_79*28ÿö@ H€'bcluster_0_1/xla_runh
5
fusion_1*28Ÿõ@  H /bcluster_2_1/xla_runh
9
fusion_32__1*28þï@ !Hà(bcluster_2_1/xla_runh
6
	fusion_35*28žæ@ Hà,bcluster_2_1/xla_runh
4
add_360*28àÝ@€!H€3bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28þ–@ H &bAssignAddVariableOp_7h
4
add_323*28ÿŒ@àH 0bcluster_2_1/xla_runh
5
slice_1*28 ñ@àH !bcluster_12_1/xla_runh
6
	fusion_26*28€ñ@ÀH 'bcluster_2_1/xla_runh
3
add_11*28þì@ Hà&bcluster_7_1/xla_runh
4
add_348*28 á@àH€!bcluster_2_1/xla_runh
4
add_335*28ŸÙ@€HÀ bcluster_2_1/xla_runh
6
	fusion_37*28¿Õ@ÀHÀ$bcluster_1_1/xla_runh
3
fusion*28ÿÓ@€H€bcluster_7_1/xla_runh
3
add_57*28àÒ@ Hàbcluster_4_1/xla_runh
6
	fusion_41*28 Ä@ HÀ#bcluster_2_1/xla_runh
3
fusion*28à´@€H€bcluster_5_1/xla_runh
3
fusion*28à¬@àH bcluster_6_1/xla_runh
5
slice_1*28€ @ÀH€bcluster_10_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€š@€H€!bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à@àHàb
LogicalAndh