
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28˜ìà@à§Húª0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ê‚½@œ™!HüÈ!bcluster_1_1/xla_runh
F
select_and_scatter_533*28ŒÂ¨@¼ƒ H¼Ø bcluster_0_1/xla_runh
9
	fusion_15*28’Îû@ÜH¼¾bcluster_0_1/xla_runh
9
	fusion_38*28±ë@œšH½‘bcluster_1_1/xla_runh
9
	fusion_11*28øóÔ@›Hýèbcluster_0_1/xla_runh
8
fusion_6*28ûïÈ@ÝÕHbcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28û‹@©H¼‡bcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28’Ý˜@ 2H ¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28Ž¶@À(H ¥bCudnnRNNh‰
9
	fusion_25*28òÿ@½õHžÕbcluster_1_1/xla_runh
9
	fusion_20*28Ô‚ê@žÔHžäbcluster_1_1/xla_runh
9
	fusion_24*28±é@¾ÊHžÙbcluster_1_1/xla_runh
9
	fusion_16*28ø‹É@žÅHžîbcluster_0_1/xla_runh
A
reduce_window_119*28ú†¦@þÚHÞ¯bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28¿êþ@ßãHÿÚbcluster_1_1/xla_runh
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ÿ‰õ@ßÇ
Hþíbcluster_0_1/xla_runh
Ã
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28¯ð@€H ^bCudnnRNNhá
9
	fusion_19*28¦ƒÜ@ÿý	Hÿû
bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28‰šÖ@€HÿgXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_23*28ÇöÌ@þ·	Hß¤
bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ÇðÈ@ß÷HßÃbcluster_0_1/xla_runh
8
reduce_4*28…çÄ@ß‰	H¿‡
bcluster_0_1/xla_runh
8
fusion_1*28‰ªÄ@¿ùHß÷	bcluster_1_1/xla_runh
8
reduce_3*28é°Ã@¿ˆ	H¿À	bcluster_0_1/xla_runh
8
reduce_5*28Êï·@¿¿Hÿ“	bcluster_0_1/xla_runh
6
reduce*28êì³@¿‘HŸÃ	bcluster_1_1/xla_runh
8
reduce_1*28ËÖ²@Ÿ–Hßœ	bcluster_1_1/xla_runh
9
	fusion_39*28––@ßçHÿÉbcluster_1_1/xla_runh
9
	fusion_23*28ÒŸ”@¿õHÿ—bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ò@¿ÕHßÖbcluster_1_1/xla_runh
8
	fusion_28*28Ðð{@ŸÞHŸbcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28Ì÷x@à/HàùXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28÷øq@à‹H¿Ùbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ï”j@ßÜHŸ±b2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28Õ¦a@ßºHŸ‡bcluster_0_1/xla_runh
E
select_and_scatter_313*28÷ÙR@ßÜH —bcluster_0_1/xla_runh
5
fusion*28ò”O@€ÑHßöbcluster_1_1/xla_runh
8
	fusion_12*28ú”H@à†H€óbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28»Œ>@àHH€Ñbcluster_0_1/xla_runh*
8
	fusion_37*28–À8@ŸÎH€ëbcluster_0_1/xla_runh
8
	fusion_41*28¼8@ÀÏHÀábcluster_0_1/xla_runh
8
	fusion_40*28ø»7@À±HÀŒbcluster_1_1/xla_runh
8
	fusion_33*28¹Ð0@ßŸH ¿bcluster_0_1/xla_runh
7
fusion_3*28œÑ*@ èHÿÉbcluster_1_1/xla_runh
8
	fusion_54*28üÜ%@àÙHàˆbcluster_0_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28œú @ H€ýb4model/dropout_2/dropout/random_uniform/RandomUniformh
E
select_and_scatter_143*28ù@ ¸H Íbcluster_0_1/xla_runh
8
	fusion_16*28ÜÊ@ ¥H€çbcluster_1_1/xla_runh
7
reduce_1*28üÂ@À¦HÀËbcluster_0_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¼Ó@ÿ¦HŸÍbtranspose_0h
8
	fusion_17*28žú@€§HàÀbcluster_1_1/xla_runh
8
	fusion_27*28Ý§@àžH€±bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾ù@ •H€b"gradients/transpose_grad/transposeh
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28½@ÀvH ²b4model/dropout_1/dropout/random_uniform/RandomUniformh
7
fusion_4*28Ûî@ €Hàšbcluster_2_1/xla_runh
@
reduce_window_193*28Ýž@ÀˆHà˜bcluster_1_1/xla_runh
7
reduce_3*28þ÷@ßHÀ¾bcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ßí@ |H€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
7
	fusion_15*28þŸ@€hHà‘bcluster_1_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28ýß@€#H€SbCudnnRNNh*
3
reduce*28ŸÞ@ÀoHÿ~bcluster_0_1/xla_runh
6
reduce_2*28üå@àmH€€bcluster_1_1/xla_runh
7
	fusion_56*28àÑ@€fH€ƒbcluster_0_1/xla_runh
6
fusion_8*28ž¬@àfH ¢bcluster_2_1/xla_runh
7
	fusion_11*28œÀ@ÿUH€—bcluster_1_1/xla_runh
5
reduce_2*28žª@€aH€gbcluster_0_1/xla_runh
5
fusion_5*28ÿñ@à]H¿fbcluster_1_1/xla_runh
3
fusion*28½º@¿VH€dbcluster_8_1/xla_runh
3
fusion*28 ˆ@ YH ^bcluster_3_1/xla_runh
5
reduce_4*28ÿþ@àVHÀebcluster_1_1/xla_runh
5
fusion_2*28¿Ó@àNH sbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾Ã@ÀKHàeb%Adam/Adam/update_12/ResourceApplyAdamh
6
	fusion_58*28¿«@ NH€lbcluster_0_1/xla_runh
5
fusion_7*28ÞÏ@àMH€`bcluster_0_1/xla_runh
>
reduce_window_263*28ŸÍ@à:HÿYbcluster_1_1/xla_runh
6
	fusion_24*28ý©@ÀLH€\bcluster_0_1/xla_runh
5
reduce_5*28¾@ÀFHàpbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ß„@àIH€ab%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_43*28üô@àKHÿRbcluster_0_1/xla_runh
6
	fusion_23*28¾°@ßFHÀPbcluster_2_1/xla_runh
6
	fusion_60*28œÂ@àCH€Jbcluster_0_1/xla_runh
4
fusion*28½¿@ÀCH€Ibcluster_10_1/xla_runh
5
fusion_9*28ý¨@€;HÀObcluster_1_1/xla_runh
3
fusion*28Ÿð
@À>HàKbcluster_9_1/xla_runh
9
fusion_33__2*28ßÄ
@à0H¿Lbcluster_2_1/xla_runh
5
fusion_8*28ž 
@à<H€Abcluster_1_1/xla_runh
6
	fusion_10*28ž÷	@€2HÀrbcluster_1_1/xla_runh
6
	fusion_26*28Àç	@À5Hà`bcluster_1_1/xla_runh
4
copy_57*28À²	@€0HÀ=bcluster_0_1/xla_runh
4
copy_57*28‰	@ -H¿dbcluster_1_1/xla_runh
6
	fusion_61*28½ð@ÿ2H¿<bcluster_0_1/xla_runh
6
	fusion_18*28ÝÇ@ /H @bcluster_1_1/xla_runh
4
add_266*28¾»@ß-H Hbcluster_2_1/xla_runh
6
	fusion_16*28€º@ 1H 7bcluster_2_1/xla_runh
6
	fusion_42*28ß«@€"HÀNbcluster_1_1/xla_runh
6
	fusion_41*28ß¢@ß1Hà4bcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Ÿ @À$H€Eb%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_30*28À™@à+H Gbcluster_2_1/xla_runh
6
	fusion_43*28€ø@ %H€:bcluster_1_1/xla_runh
4
copy_50*28ßò@À*H€<bcluster_1_1/xla_runh
3
fusion*28 Ñ@€)H€:bcluster_2_1/xla_runh
5
fusion_6*28Ÿ¬@à*Hà/bcluster_4_1/xla_runh
5
fusion_6*28àþ@ÀHÀ6bcluster_2_1/xla_runh
4
copy_72*28ßÜ@¿'H .bcluster_0_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 ¬@À#H€+b
div_no_nanh
6
	fusion_33*28ý¥@ŸHÀ1bcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28¿Š@à!Hà-bcluster_1_1/xla_runh
5
fusion_1*28ÿ‡@à Hà+bcluster_2_1/xla_runh
9
fusion_33__1*28ß…@€H€,bcluster_2_1/xla_runh
6
	fusion_36*28Ÿ…@àHÀ)bcluster_2_1/xla_runh
6
	fusion_21*28Ýõ@€ H€)bcluster_2_1/xla_runh
6
	fusion_48*28Ÿò@  Hà+bcluster_2_1/xla_runh
6
	fusion_50*28€Ø@ÀH€,bcluster_0_1/xla_runh
4
add_368*28 »@À HÀ"bcluster_2_1/xla_runh
6
	fusion_49*28À¶@ÀHÀ$bcluster_2_1/xla_runh
6
	fusion_44*28À³@ HÀ,bcluster_1_1/xla_runh
6
	fusion_45*28À¦@€H 5bcluster_1_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ß@ÀHß"bAssignAddVariableOp_7h
4
add_343*28Àû@€HÀ$bcluster_2_1/xla_runh
6
	fusion_42*28àø@ H€%bcluster_2_1/xla_runh
4
add_356*28¿õ@€HÀ bcluster_2_1/xla_runh
4
add_331*28Àã@ÀHÀ bcluster_2_1/xla_runh
4
slice_1*28 ×@€Hàbcluster_9_1/xla_runh
6
	fusion_27*28ÿÖ@€Hàbcluster_2_1/xla_runh
3
add_39*28àÐ@€Hàbcluster_4_1/xla_runh
3
fusion*28À«@àH€bcluster_7_1/xla_runh
3
fusion*28à@àH€bcluster_5_1/xla_runh
3
add_11*28¿™@àHàbcluster_7_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à‚@€HàbAssignAddVariableOp_1h
3
fusion*28Þú@ÀHÀbcluster_6_1/xla_runh
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À!@À!HÀ!b
LogicalAndh