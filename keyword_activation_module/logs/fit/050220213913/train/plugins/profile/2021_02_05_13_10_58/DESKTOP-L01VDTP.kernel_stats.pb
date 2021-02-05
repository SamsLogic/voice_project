

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ýÜ•@½íH™ÒDbcluster_0_1/xla_runh*
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Æ÷€@Ý­Hú¯0bcluster_0_1/xla_runh*
F
select_and_scatter_533*28ÇŒò
@¸ÿAH¸áBbcluster_0_1/xla_runh
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28êÙ
@ØHœ§(bcluster_1_1/xla_runh*
9
	fusion_11*28¸ëÂ	@ùê9Hú¤:bcluster_0_1/xla_runh
9
	fusion_15*28µ·»	@º¡9Hù‘:bcluster_0_1/xla_runh
9
	fusion_38*28ú©¹	@š§9Hú„:bcluster_1_1/xla_runh
8
fusion_6*28ºó´	@ù”9HšÌ9bcluster_0_1/xla_runh
9
	fusion_25*28ˆœ@œƒ%H¼Ö%bcluster_1_1/xla_runh
9
	fusion_20*28ìðù@Ü¦#Hœý$bcluster_1_1/xla_runh
9
	fusion_24*28ŒÓø@üÃ#HÜÚ$bcluster_1_1/xla_runh
:
sgemm_32x32x32_NN_vec*28ÌêÕ@à)Hà¯bCudnnRNNhê

\
sgemm_32x32x32_NN_vec*28Ð¯Ó@€1H€¸Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophê

£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ù¨‡@ÝÌHÜŸ%bcluster_0_1/xla_runh
9
	fusion_16*28øÄÿ@Ý”H¼‡bcluster_0_1/xla_runh
A
reduce_window_119*28ŽÏ@ýæHý¿bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28à¢¿@ÄHýbcluster_1_1/xla_runh
9
	fusion_19*28Òã­@¾HÞ÷bcluster_1_1/xla_runh
Ä
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28Õõ©@àH€ŠbCudnnRNNhÄ
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28˜ß’@€HàtXb(gradients/CudnnRNN_grad/CudnnRNNBackprophÄ
8
reduce_4*28µš@þÛHž´bcluster_0_1/xla_runh
8
reduce_1*28¶ÿ‹@ž÷HÞùbcluster_1_1/xla_runh
8
fusion_1*28õŠ@¾¶H¾—bcluster_1_1/xla_runh
6
reduce*28Öãƒ@žûHž”bcluster_1_1/xla_runh
8
reduce_5*28Ö–ÿ@žýHž¿bcluster_0_1/xla_runh
8
reduce_3*28Õªü@ÞíHÝËbcluster_0_1/xla_runh
9
	fusion_32*28›úª@¾HÞÐbcluster_0_1/xla_runh
9
	fusion_39*28€ ª@¿ëHÿÊbcluster_1_1/xla_runh
9
	fusion_28*28ÞÊž@þ¼HÞîbcluster_0_1/xla_runh
F
select_and_scatter_313*28åÆ–@Ÿ†HßÎbcluster_0_1/xla_runh
9
	fusion_23*28¾”@¿âH¾bcluster_0_1/xla_runh
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28„ª@ß…HßÚb2model/dropout/dropout/random_uniform/RandomUniformh
9
	fusion_23*28Èÿ÷@Ÿ»Hþ¤bcluster_1_1/xla_runh
6
fusion*28¥°ò@þHÞ‡bcluster_1_1/xla_runh
[
sgemm_32x32x32_TN_vec*28ªƒÞ@à-HÿôXb(gradients/CudnnRNN_grad/CudnnRNNBackproph~
w
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28Ì†Ä@€zHÿèbcluster_0_1/xla_runh*
9
	fusion_37*28«¾À@ß·H¿‚bcluster_0_1/xla_runh
9
	fusion_47*28ÎÖ©@ÀàHÿ»bcluster_0_1/xla_runh
9
	fusion_17*28²®¨@àÚHŸ¨bcluster_1_1/xla_runh
9
	fusion_12*28Ïž¤@Ÿ¶H¿³bcluster_1_1/xla_runh
9
	fusion_33*28®½š@ÿŽHŸÐbcluster_0_1/xla_runh
9
	fusion_41*28Î‘—@ÿáHÿ‘bcluster_0_1/xla_runh
9
	fusion_16*28®µ‡@ÿƒHŸÔbcluster_1_1/xla_runh
7
fusion_3*28“£c@ÿ¤H¿€bcluster_1_1/xla_runh
@
reduce_window_193*28°Ùa@ÿªHßúbcluster_1_1/xla_runh
7
fusion_5*28·öY@ÀðH¿Ébcluster_1_1/xla_runh
7
reduce_1*28º€R@€ÚHà¤bcluster_0_1/xla_runh
8
	fusion_56*28ûôN@ÿËHÀŠbcluster_0_1/xla_runh
8
	fusion_40*28µÕM@ ”HŸ†bcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ÞF@ÀæH Ïb4model/dropout_1/dropout/random_uniform/RandomUniformh
7
reduce_3*28ÜºE@€êHŸ¥bcluster_1_1/xla_runh
5
reduce*28Ú”C@àõHàÂbcluster_0_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28öäA@¿|HÀ¡btranspose_0h*
7
reduce_2*28½žA@àùHà¤bcluster_0_1/xla_runh
8
	fusion_15*28üï@@àøHÀ›bcluster_1_1/xla_runh
7
reduce_2*28˜‹@@ÿäHŸªbcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ºÌ>@ †HÀˆb"gradients/transpose_grad/transposeh*
E
select_and_scatter_143*28œ·<@ÀÔH bcluster_0_1/xla_runh
7
fusion_2*28º‘7@À H Šbcluster_1_1/xla_runh
8
	fusion_27*28¼4@€¡Hÿòbcluster_1_1/xla_runh
8
	fusion_54*28˜¶1@ ŠHÀÒbcluster_0_1/xla_runh
8
	fusion_11*28úÐ/@à÷HàÇbcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ºî)@àáHÀÙb4model/dropout_2/dropout/random_uniform/RandomUniformh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ú¯(@àmHÀˆXb(gradients/CudnnRNN_grad/CudnnRNNBackproph*
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28Üò'@ "H€ibCudnnRNNhT
7
fusion_9*28úè&@ßÃHÀbcluster_1_1/xla_runh
8
	fusion_10*28öÐ%@ ÒHßŠbcluster_1_1/xla_runh
6
copy_50*28à$@€ÄH þbcluster_1_1/xla_runh
6
copy_57*28Ýý @à¶HÀøbcluster_0_1/xla_runh
8
	fusion_58*28þ@ßƒH ¨bcluster_0_1/xla_runh
7
reduce_5*28€Ì@àˆH •bcluster_1_1/xla_runh
?
reduce_window_263*28½Ž@ ]HàÉbcluster_1_1/xla_runh
6
fusion_4*28½ø@à|Hß¤bcluster_2_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Þù@À‚H Šbtranspose_9h

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28þÔ@ÀqH€‘b%Adam/Adam/update_12/ResourceApplyAdamh
´
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¿à@À{HÀƒb$gradients/transpose_9_grad/transposeh
6
reduce_4*28þ”@ sH€™bcluster_1_1/xla_runh
6
fusion_8*28Àÿ@ ZH Šbcluster_1_1/xla_runh
6
fusion_8*28ÞÉ@€aH¿—bcluster_2_1/xla_runh
3
fusion*28ž”@€cHÀsbcluster_8_1/xla_runh
¬
ƒ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28þ®@¿_HÀgbgradients/AddNh
6
	fusion_18*28žÌ@€NHŸsbcluster_1_1/xla_runh
4
fusion*28œº@€EH ‡bcluster_9_1/xla_runh
3
fusion*28ž‡@ YHà_bcluster_3_1/xla_runh
5
fusion_7*28ý¡@ LHÀfbcluster_0_1/xla_runh
6
	fusion_24*28žò@€MHàibcluster_0_1/xla_runh
6
	fusion_43*28¿à@€QHàVbcluster_0_1/xla_runh
4
copy_57*28ÞÀ@€OHÀXbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Þ¨@ÀKH¿bb%Adam/Adam/update_13/ResourceApplyAdamh
4
copy_72*28ß’@ÀKHŸ^bcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾þ@ÀBHàcb%Adam/Adam/update_15/ResourceApplyAdamh
7
	fusion_41*28ž«@€DHàbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ß‚@àAHÀTb%Adam/Adam/update_16/ResourceApplyAdamh
4
fusion*28àØ@À@H€Obcluster_10_1/xla_runh
6
	fusion_23*28 @ BH Ibcluster_2_1/xla_runh
4
fusion*28¿ñ
@à?HÿKbcluster_11_1/xla_runh
6
	fusion_60*28¿ñ
@À@HàDbcluster_0_1/xla_runh
6
	fusion_45*28ßà
@ÿ<HàTbcluster_1_1/xla_runh
9
fusion_32__2*28Ý­
@À0H Hbcluster_2_1/xla_runh
6
fusion_1*28ÿŸ
@à6HàHbcluster_12_1/xla_runh
6
	fusion_44*28àš
@À:H€cbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ÀÍ	@À%H€Eb%Adam/Adam/update_17/ResourceApplyAdamh
4
fusion*28þ½	@à.HßEbcluster_12_1/xla_runh
6
	fusion_29*28¿ž	@à0H€Wbcluster_2_1/xla_runh
6
	fusion_42*28ž•	@€-H @bcluster_1_1/xla_runh
6
	fusion_43*28¿Œ	@ß4HÀ:bcluster_1_1/xla_runh
4
add_258*28žð@À+H Xbcluster_2_1/xla_runh
6
	fusion_16*28þ¶@à/H€9bcluster_2_1/xla_runh
3
fusion*28¿©@À'HÀ<bcluster_2_1/xla_runh
5
fusion_6*28ÿ@ ,Hà7bcluster_2_1/xla_runh
6
	fusion_61*28€‰@À)HàAbcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28à¶@À$HàAb%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_32*28Þ¢@€(HßKbcluster_2_1/xla_runh
6
	fusion_47*28à«@ "H€/bcluster_2_1/xla_runh
9
fusion_32__1*28à¨@À$HÀ,bcluster_2_1/xla_runh
6
	fusion_26*28À˜@€"Hà2bcluster_1_1/xla_runh
5
fusion_6*28 ˜@€$H€)bcluster_4_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ÿ“@€#HÀ7bcluster_1_1/xla_runh
6
	fusion_21*28€‰@À Hà'bcluster_2_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ÿ„@ "H *b
div_no_nanh
6
	fusion_48*28Àƒ@à!HÀ-bcluster_2_1/xla_runh
4
add_360*28žƒ@à HÀ2bcluster_2_1/xla_runh
5
fusion_1*28ßû@À HÀ+bcluster_2_1/xla_runh
6
	fusion_35*28¿ê@  H€?bcluster_2_1/xla_runh
4
add_323*28€Ù@ HÀ=bcluster_2_1/xla_runh
6
	fusion_50*28ŸÓ@àHÀ'bcluster_0_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€¥@àH€(bAssignAddVariableOp_7h
4
add_348*28 ’@àH -bcluster_2_1/xla_runh
6
	fusion_41*28 ‘@ÀH ,bcluster_2_1/xla_runh
3
add_39*28À@ Hà!bcluster_4_1/xla_runh
5
slice_1*28 ý@€Hà bcluster_12_1/xla_runh
6
	fusion_26*28àì@àH &bcluster_2_1/xla_runh
4
add_335*28Àà@àH€!bcluster_2_1/xla_runh
3
add_11*28à·@€H 5bcluster_7_1/xla_runh
5
slice_1*28Þ¨@€HÀ,bcluster_10_1/xla_runh
3
fusion*28à¤@€HÀbcluster_5_1/xla_runh
3
fusion*28à¡@ Hàbcluster_6_1/xla_runh
3
fusion*28ß @€H bcluster_7_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€‰@€HÀ bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À!@À!HÀ!b
LogicalAndh