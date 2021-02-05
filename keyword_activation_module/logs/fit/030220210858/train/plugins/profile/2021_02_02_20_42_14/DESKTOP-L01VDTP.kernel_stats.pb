
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28èÛã@ÿ°HûÁ0bcluster_1_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28´Íº@½!H¼Ç!bcluster_2_1/xla_runh
F
select_and_scatter_533*28“—©@¼‘ Hœï bcluster_1_1/xla_runh
9
	fusion_15*28šÑû@½ˆH½½bcluster_1_1/xla_runh
9
	fusion_38*28ÝÜê@”H¼øbcluster_2_1/xla_runh
9
	fusion_11*28üþÓ@žHÝÄbcluster_1_1/xla_runh
8
fusion_6*28àØÌ@ýÞHÝ¤bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ˆë”@ŸH¾übcluster_1_1/xla_runh
\
sgemm_32x32x32_NN_vec*28“Ö¿@À2H€ÒXb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28ÉÅ@€(H ¥bCudnnRNNh‰
9
	fusion_25*28–µý@žíH¾Ãbcluster_2_1/xla_runh
9
	fusion_24*28ºŒç@þµHÞÀbcluster_2_1/xla_runh
9
	fusion_20*28¹åæ@žÈHþÃbcluster_2_1/xla_runh
9
	fusion_16*28ýžË@ß½HÞìbcluster_1_1/xla_runh
A
reduce_window_119*28Ú¤@žÐHþ¼bcluster_2_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28Äëý@žßHÿ·bcluster_2_1/xla_runh
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28æö@ÿÂ
HŸÐbcluster_1_1/xla_runh
Ä
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28£Ùó@àHà¤bCudnnRNNhá
²
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28Äáï@€HÀ½Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_19*28‹ÑÝ@¿‹
Hßbcluster_2_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ÅÚÓ@ÿ	HþÙbcluster_1_1/xla_runh
9
	fusion_23*28©ËÏ@ŸÄ	H¿Ë
bcluster_2_1/xla_runh
8
reduce_4*28«‹Ä@ÿƒ	H¿÷	bcluster_1_1/xla_runh
8
reduce_3*28‹¦Ã@ß„	H¿Ñ	bcluster_1_1/xla_runh
8
fusion_1*28«ÀÂ@¿õHß¾	bcluster_2_1/xla_runh
8
reduce_1*28Œüº@ÿžHŸ—
bcluster_2_1/xla_runh
8
reduce_5*28íâ¹@€ÍHÿ«	bcluster_1_1/xla_runh
6
reduce*28®„¹@ÀŽH¿ß	bcluster_2_1/xla_runh
9
	fusion_39*28¯Þ•@ÿÛHÿ±bcluster_2_1/xla_runh
9
	fusion_23*28Ò ”@ÿøHÿ¥bcluster_1_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28î}@ÿ×Hÿ­bcluster_2_1/xla_runh
8
	fusion_28*28ôœ{@ÀÍH€—bcluster_1_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28²Àx@ -HÀ‚Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28¯úq@€™HŸÑbcluster_1_1/xla_runh
8
	fusion_32*28™Ôa@à¼H ìbcluster_1_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¶çS@ÀðH Ñb2model/dropout/dropout/random_uniform/RandomUniformh
E
select_and_scatter_313*28óÚR@À×HŸ›bcluster_1_1/xla_runh
5
fusion*28”„R@¿ÓHß¢bcluster_2_1/xla_runh
8
	fusion_12*28³¸F@ßúHÿìbcluster_2_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28”ñA@ GHÀÛbcluster_1_1/xla_runh*
8
	fusion_37*28¹«8@ßÍH€èbcluster_1_1/xla_runh
8
	fusion_41*28×š8@ ÐH€àbcluster_1_1/xla_runh
8
	fusion_40*28ù»6@ ¯Hßébcluster_2_1/xla_runh
8
	fusion_33*28úÖ1@àH€Èbcluster_1_1/xla_runh
7
fusion_3*28›”+@€íHà¬bcluster_2_1/xla_runh
8
	fusion_54*28ü‚)@ÀáH€bcluster_1_1/xla_runh
7
	fusion_15*28 "@€kH€“bcluster_2_1/xla_runh
8
	fusion_16*28œì @àªH¿÷bcluster_2_1/xla_runh
E
select_and_scatter_143*28þÏ @à·HÿÖbcluster_1_1/xla_runh
7
reduce_1*28Ú¯@ ¤H¿Ñbcluster_1_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾Û@À¡HŸÎbtranspose_0h
8
	fusion_17*28ß’@€¢H€¹bcluster_2_1/xla_runh
8
	fusion_27*28Ý¾@àHà»bcluster_2_1/xla_runh
6
reduce_3*28žœ@Ÿ}H€Ñbcluster_2_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28›Ð@€’Hÿšb"gradients/transpose_grad/transposeh
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ŸÀ@ |Hà¨b4model/dropout_1/dropout/random_uniform/RandomUniformh
7
fusion_4*28þã@ €H€˜bcluster_3_1/xla_runh
@
reduce_window_193*28À¿@€†HàÄbcluster_2_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28¾Ì@à~H€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
4
reduce*28ÜÍ@àsH ‘bcluster_1_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28þÏ@€"H€SbCudnnRNNh*
7
	fusion_56*28¾â@€hHÿbcluster_1_1/xla_runh
7
	fusion_11*28ÝÕ@€dH »bcluster_3_1/xla_runh
5
reduce_2*28Þ½@àhH€bcluster_2_1/xla_runh
7
	fusion_11*28žö@àYHÀŽbcluster_2_1/xla_runh
5
reduce_2*28¾×@ÀbH¿xbcluster_1_1/xla_runh
5
fusion_5*28Ý”@Ÿ^H kbcluster_2_1/xla_runh
5
fusion_6*28¾Ï@à]H abcluster_0_1/xla_runh
7
	fusion_58*28´@ŸLHÀ‡bcluster_1_1/xla_runh
3
fusion*28ß•@€YH abcluster_4_1/xla_runh
5
fusion_2*28ÿè@€RHŸvbcluster_2_1/xla_runh
5
reduce_4*28¾Ý@ RHÀobcluster_2_1/xla_runh
5
reduce_5*28ý×@¿GH€ubcluster_2_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¿Ç@ WHà[b4model/dropout_2/dropout/random_uniform/RandomUniformh
>
reduce_window_263*28€…@à;Hàwbcluster_2_1/xla_runh
6
	fusion_24*28Þ¬@ÿJH€[bcluster_1_1/xla_runh
5
fusion_7*28¾¡@ MH€[bcluster_1_1/xla_runh
6
	fusion_43*28ÿø@€KHàTbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ŸÙ@€HHàib%Adam/Adam/update_12/ResourceApplyAdamh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¿Ó@ GHÀ\b%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_23*28Þ¯@ÀGHàRbcluster_3_1/xla_runh
3
fusion*28¾Ö@ÿCHÀLbcluster_6_1/xla_runh
5
fusion_9*28ÀÔ@À>HàObcluster_2_1/xla_runh
3
fusion*28Ý @ BHÀIbcluster_5_1/xla_runh
6
	fusion_60*28ýˆ@à@H Gbcluster_1_1/xla_runh
5
fusion_8*28ÿ±
@€;H _bcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 œ
@ <HÀDb%Adam/Adam/update_14/ResourceApplyAdamh
9
fusion_33__2*28 ý	@à/H Dbcluster_3_1/xla_runh
6
	fusion_26*28Ýö	@ 5Hàbbcluster_2_1/xla_runh
6
	fusion_18*28þÞ	@à/Hàlbcluster_2_1/xla_runh
4
copy_57*28€Ò	@ /H€Xbcluster_1_1/xla_runh
4
copy_50*28¿Ë	@à,Hàgbcluster_2_1/xla_runh
6
	fusion_10*28 ¢	@À/HÀibcluster_2_1/xla_runh
4
copy_57*28Ýæ@ÿ-H [bcluster_2_1/xla_runh
6
	fusion_41*28žá@€0H€tbcluster_2_1/xla_runh
6
	fusion_16*28ßÍ@€2H€=bcluster_3_1/xla_runh
6
	fusion_42*28Þ¿@à HàBbcluster_2_1/xla_runh
6
	fusion_61*28€œ@À/Hà7bcluster_1_1/xla_runh
4
add_297*28à˜@à,H€Hbcluster_3_1/xla_runh
6
	fusion_43*28 ˜@€$H Kbcluster_2_1/xla_runh
6
	fusion_30*28Þ”@ -Hà6bcluster_3_1/xla_runh
4
copy_72*28ßê@À)H€Gbcluster_1_1/xla_runh
3
fusion*28à«@à&Hà:bcluster_3_1/xla_runh
6
	fusion_10*28ÿù@à"Hà7bcluster_3_1/xla_runh
6
	fusion_33*28€Ê@À!H€5bcluster_3_1/xla_runh
6
	fusion_12*28ÿÆ@à&H€+bcluster_0_1/xla_runh
9
fusion_33__1*28@à H ,bcluster_3_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ÞŒ@¿"Hà5bcluster_2_1/xla_runh
6
	fusion_50*28ý÷@¿"HÀ(bcluster_1_1/xla_runh
6
	fusion_36*28Àí@€ H€*bcluster_3_1/xla_runh
5
fusion_1*28€ì@àHÀ.bcluster_3_1/xla_runh
3
add_38*28ÿè@€"H€&bcluster_0_1/xla_runh
6
	fusion_21*28 á@  H€(bcluster_3_1/xla_runh
3
add_70*28ÿÓ@à H 0bcluster_0_1/xla_runh
6
	fusion_11*28¾Ð@àH (bcluster_0_1/xla_runh
4
add_350*28¿Á@  H€(bcluster_3_1/xla_runh
6
	fusion_13*28Àµ@ HÀ$bcluster_0_1/xla_runh
6
	fusion_42*28€©@ Hà0bcluster_3_1/xla_runh
6
	fusion_49*28ÿ¨@ÀH€#bcluster_3_1/xla_runh
6
	fusion_44*28þ¦@àHÀ)bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Àœ@ H€$bAssignAddVariableOp_7h
6
	fusion_45*28€œ@ÀH€,bcluster_2_1/xla_runh
4
add_375*28Àí@€Hà bcluster_3_1/xla_runh
4
add_338*28Àç@àHà$bcluster_3_1/xla_runh
6
	fusion_47*28¿æ@àHàbcluster_3_1/xla_runh
4
add_363*28ÀÜ@àH€%bcluster_3_1/xla_runh
4
slice_1*28Ÿ´@ Hà bcluster_5_1/xla_runh
6
	fusion_27*28À§@ H &bcluster_3_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÀŽ@€Hà bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€@€H€b
LogicalAndh