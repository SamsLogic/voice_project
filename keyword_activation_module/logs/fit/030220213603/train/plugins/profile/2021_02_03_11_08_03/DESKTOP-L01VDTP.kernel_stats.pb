
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28œöâ@¿©HÚÕ1bcluster_1_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Òü¹@Üù HÜØ!bcluster_2_1/xla_runh
F
select_and_scatter_533*28”ý¨@“ H¼Ü bcluster_1_1/xla_runh
9
	fusion_15*28öÁû@ÜŠHÝ¹bcluster_1_1/xla_runh
9
	fusion_38*28º¼ë@ýHëbcluster_2_1/xla_runh
9
	fusion_11*28›øÓ@¼‘HüÚbcluster_1_1/xla_runh
8
fusion_6*28ß˜Ê@ÝÞH½‘bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ã»•@ýàHšbcluster_1_1/xla_runh
\
sgemm_32x32x32_NN_vec*28í•³@ 2H€ºXb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28êô‘@ß)H€¥bCudnnRNNh‰
9
	fusion_25*28Ó•þ@þçHþÊbcluster_2_1/xla_runh
9
	fusion_20*28ö‘é@¾¾HÞÙbcluster_2_1/xla_runh
9
	fusion_24*28÷°ç@Þ¿HÞÍbcluster_2_1/xla_runh
9
	fusion_16*28¹ªÊ@þÁHž—bcluster_1_1/xla_runh
A
reduce_window_119*28ÿÝ¥@¿ÂH¿±bcluster_2_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28â›€@ßòHþÐbcluster_2_1/xla_runh
Ä
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28Æ’õ@€HÀÀbCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28££ô@ŸÉ
HÞ¿bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28†±è@€HàtXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_19*28èÌÜ@¿Ž
Hþò
bcluster_2_1/xla_runh
9
	fusion_23*28ÆïÐ@ŸÆ	H¾Ë
bcluster_2_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Ë¤É@¿óHÿ¨bcluster_1_1/xla_runh
8
reduce_4*28ˆçÄ@ß‰	H¿í	bcluster_1_1/xla_runh
8
reduce_3*28è…Ä@ÿ	HŸë	bcluster_1_1/xla_runh
8
fusion_1*28«Â¿@ÿêHÿÒ	bcluster_2_1/xla_runh
6
reduce*28ë†¸@¿‡H¿›
bcluster_2_1/xla_runh
8
reduce_5*28ëÝ·@ŸÂHß 	bcluster_1_1/xla_runh
8
reduce_1*28ËÇ¶@ß˜Hž•
bcluster_2_1/xla_runh
9
	fusion_39*28ðÑ–@¿íHàÌbcluster_2_1/xla_runh
9
	fusion_23*28ð«”@àúHŸ¥bcluster_1_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ù~@ŸÓH¿Èbcluster_2_1/xla_runh
8
	fusion_28*28“õz@ ÒH ‹bcluster_1_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28òáy@€-HÀüXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28ôÏq@ÿ˜HÿÜbcluster_1_1/xla_runh
8
	fusion_32*28´¨c@ŸÇHàˆbcluster_1_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¸¸`@€ÙHŸ€b2model/dropout/dropout/random_uniform/RandomUniformh
E
select_and_scatter_313*28×óS@€ìH¿¼bcluster_1_1/xla_runh
5
fusion*28÷ƒQ@€ÑH¿£bcluster_2_1/xla_runh
8
	fusion_12*28Ö©G@ÀˆHàùbcluster_2_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28¼A@€HHÀÒbcluster_1_1/xla_runh*
8
	fusion_41*28›Î8@ÿÐHàâbcluster_1_1/xla_runh
8
	fusion_37*28Ú8@ÀÊH ëbcluster_1_1/xla_runh
8
	fusion_40*28Û‹6@ß©Hàëbcluster_2_1/xla_runh
8
	fusion_33*28›ù0@ÿHŸçbcluster_1_1/xla_runh
7
fusion_3*28š•*@¿ìH šbcluster_2_1/xla_runh
8
	fusion_54*28Ù“(@ÀÔH¿²bcluster_1_1/xla_runh
8
	fusion_15*28Ù‡(@ ÞHßŸbcluster_2_1/xla_runh
E
select_and_scatter_143*28þ!@Ÿ¸Hàßbcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ï@à”H€çb4model/dropout_1/dropout/random_uniform/RandomUniformh
7
reduce_1*28â@€¨HÿÈbcluster_1_1/xla_runh
8
	fusion_16*28­@€£H ×bcluster_2_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ùæ@€¡H Äbtranspose_0h
8
	fusion_17*28¼Ï@¿žHß»bcluster_2_1/xla_runh
8
	fusion_27*28¾€@ ‰Hà­bcluster_2_1/xla_runh
7
reduce_3*28¿à@à‚Hà×bcluster_2_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28»Ô@à’Hÿ¥b"gradients/transpose_grad/transposeh
6
fusion_4*28Ü¯@Ÿ|HÀ›bcluster_3_1/xla_runh
?
reduce_window_193*28þ¨@ÀHÀ¤bcluster_2_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ßÅ@À~H€ŠXb(gradients/CudnnRNN_grad/CudnnRNNBackproph

j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28Û¥@À"Hà‚bCudnnRNNh*
4
reduce*28ÞÛ@€iH ‡bcluster_1_1/xla_runh
7
	fusion_11*28ÿ†@àaHÀ¾bcluster_3_1/xla_runh
5
reduce_2*28¿…@€mHÀ~bcluster_2_1/xla_runh
7
	fusion_56*28ÿ”@€dH –bcluster_1_1/xla_runh
7
	fusion_11*28Þà@ÀSHÿbcluster_2_1/xla_runh
5
reduce_2*28ÞÒ@ÀcHàlbcluster_1_1/xla_runh
5
fusion_6*28Þì@€^H hbcluster_0_1/xla_runh
5
fusion_5*28Àì@ OH€kbcluster_2_1/xla_runh
6
fusion_2*28€Ó@€PH€bcluster_2_1/xla_runh
6
	fusion_58*28Ü§@ÀOHŸkbcluster_1_1/xla_runh
3
fusion*28ß†@àSHàbbcluster_4_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ßõ@€YHà]b4model/dropout_2/dropout/random_uniform/RandomUniformh
5
reduce_4*28ÞÅ@àQH bbcluster_2_1/xla_runh
5
fusion_7*28ß„@€OH€_bcluster_1_1/xla_runh
>
reduce_window_263*28¿ó@à;HàXbcluster_2_1/xla_runh
6
	fusion_24*28 Û@€LHÀ]bcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 É@ JHÀub%Adam/Adam/update_12/ResourceApplyAdamh
6
	fusion_43*28ž£@¿LH€Ubcluster_1_1/xla_runh
6
	fusion_23*28ží@ KHàRbcluster_3_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ÿå@àFH Wb%Adam/Adam/update_13/ResourceApplyAdamh
5
reduce_5*28àƒ@ÀEH nbcluster_2_1/xla_runh
3
fusion*28ßË@àCHàKbcluster_6_1/xla_runh
6
	fusion_60*28 Ë@€CH Ibcluster_1_1/xla_runh
5
fusion_9*28Àº@À<HàNbcluster_2_1/xla_runh
3
fusion*28ÿÕ
@À=H Ebcluster_5_1/xla_runh
6
	fusion_10*28€’
@€1H pbcluster_2_1/xla_runh
5
fusion_8*28žï	@À:H Abcluster_2_1/xla_runh
6
	fusion_26*28Þî	@À7HÀVbcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¿å	@à$H Hb%Adam/Adam/update_14/ResourceApplyAdamh
4
copy_57*28àÁ	@ 0HàCbcluster_1_1/xla_runh
9
fusion_33__2*28€¼	@ .H Fbcluster_3_1/xla_runh
6
	fusion_18*28 ñ@à.HàBbcluster_2_1/xla_runh
6
	fusion_61*28àç@À3Hà:bcluster_1_1/xla_runh
4
add_297*28ÀÎ@À,H€Xbcluster_3_1/xla_runh
6
	fusion_16*28ŸÎ@À2Hÿ8bcluster_3_1/xla_runh
6
	fusion_42*28¿º@À!H€?bcluster_2_1/xla_runh
4
copy_50*28¿°@à*HÀabcluster_2_1/xla_runh
4
copy_57*28¿¦@Ÿ+HÀFbcluster_2_1/xla_runh
6
	fusion_41*28¿¦@à1H€5bcluster_2_1/xla_runh
6
	fusion_43*28à¢@à$HÀYbcluster_2_1/xla_runh
6
	fusion_30*28ý’@À-H€Kbcluster_3_1/xla_runh
3
fusion*28ÿð@€)H€:bcluster_3_1/xla_runh
6
	fusion_10*28ÀÖ@ #HàWbcluster_3_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28Ÿæ@€$H Dbcluster_2_1/xla_runh
4
copy_72*28þà@à&HÀ1bcluster_1_1/xla_runh
6
	fusion_33*28 Ð@à!Hà1bcluster_3_1/xla_runh
5
fusion_1*28¿¶@ß H .bcluster_3_1/xla_runh
6
	fusion_11*28¿´@ "H€Bbcluster_0_1/xla_runh
6
	fusion_12*28à²@à#H *bcluster_0_1/xla_runh
6
	fusion_21*28¾°@ "H€-bcluster_3_1/xla_runh
6
	fusion_50*28€‹@ "HÀ9bcluster_1_1/xla_runh
9
fusion_33__1*28ž‚@à!Hà+bcluster_3_1/xla_runh
6
	fusion_13*28ÿý@ÿ!H€/bcluster_0_1/xla_runh
6
	fusion_36*28ÿé@€HÀ'bcluster_3_1/xla_runh
3
add_38*28ßß@à!HÀ%bcluster_0_1/xla_runh
6
	fusion_45*28ÿÅ@ÀHÀ5bcluster_2_1/xla_runh
3
add_70*28ÀÀ@à H #bcluster_0_1/xla_runh
4
add_350*28À¹@àH #bcluster_3_1/xla_runh
6
	fusion_49*28À¶@àH€$bcluster_3_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ß¡@ HÀ,bAssignAddVariableOp_7h
4
add_338*28€„@àHÀ+bcluster_3_1/xla_runh
6
	fusion_44*28Àþ@ÀHÀ bcluster_2_1/xla_runh
4
add_363*28þð@àH€'bcluster_3_1/xla_runh
6
	fusion_47*28ÿë@ÀH€#bcluster_3_1/xla_runh
6
	fusion_42*28þê@ H %bcluster_3_1/xla_runh
4
add_375*28€Ü@àH€,bcluster_3_1/xla_runh
4
slice_1*28ß²@àH€!bcluster_5_1/xla_runh
6
	fusion_27*28ßŽ@ HÀbcluster_3_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À†@€HÀbAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À@ÀHÀb
LogicalAndh