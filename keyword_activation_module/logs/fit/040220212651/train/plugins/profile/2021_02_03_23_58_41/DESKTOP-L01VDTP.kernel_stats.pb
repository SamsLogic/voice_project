
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ãİ@ÿ¨HÛ¶0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28¬ë»@ü‘!Hü¿!bcluster_1_1/xla_runh
F
select_and_scatter_533*28íÙª@üš HÜú bcluster_0_1/xla_runh
9
	fusion_15*28Ø·û@ı‡Hœ¶bcluster_0_1/xla_runh
9
	fusion_38*28Üˆë@H½ébcluster_1_1/xla_runh
9
	fusion_11*28ùğÔ@ıHÜÓbcluster_0_1/xla_runh
8
fusion_6*28œ«Ë@ıãHbcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28á›@ı­Hİºbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28ƒ–@€2Hÿ¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28¸¦@à(HÀ¢bCudnnRNNh‰
9
	fusion_25*28ÏÊı@ıçHÅbcluster_1_1/xla_runh
9
	fusion_24*28õÀé@¾ÇHÚbcluster_1_1/xla_runh
9
	fusion_20*28öùè@¾¸HÊbcluster_1_1/xla_runh
9
	fusion_16*28¸©É@ÂHşíbcluster_0_1/xla_runh
A
reduce_window_119*28¹™¤@ÎHŞ·bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28¡ı@¾ãHŞÌbcluster_1_1/xla_runh
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28¤›î@ŸÅ
H¾Îbcluster_0_1/xla_runh
Ã
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28£Îí@€H€\bCudnnRNNhá
9
	fusion_19*28‰–Ü@ß‹
H¿ø
bcluster_1_1/xla_runh
9
	fusion_23*28¤´Ï@ÿ·	Hÿµ
bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ìÍ@€HÀjXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
8
reduce_4*28éïÄ@Ÿ—	HŸß	bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28§¥Ä@ßğHÿÈ
bcluster_0_1/xla_runh
8
reduce_3*28¨ŸÄ@ßˆ	Hßğ	bcluster_0_1/xla_runh
8
fusion_1*28§“Â@ÿHÿÕ	bcluster_1_1/xla_runh
8
reduce_5*28ÈÈ¸@ß¿HşÕ	bcluster_0_1/xla_runh
6
reduce*28‹†µ@¿•HŸÃ	bcluster_1_1/xla_runh
8
reduce_1*28ÊÚ´@¿HŞô	bcluster_1_1/xla_runh
9
	fusion_39*28±÷–@ÿÜHàĞbcluster_1_1/xla_runh
9
	fusion_23*28­£•@ŸüHÿbcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ñ–}@¿ÚHà¯bcluster_1_1/xla_runh
8
	fusion_28*28±×{@ àHàbcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28ğÕx@à*H¿öXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28ÑÚr@ÿ”H¿ßbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28µÿd@ŸÓHà†b2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28Ó¨`@À¹Hÿâbcluster_0_1/xla_runh
E
select_and_scatter_313*28—œR@€ÜH •bcluster_0_1/xla_runh
5
fusion*28ûìO@àÔHàbcluster_1_1/xla_runh
8
	fusion_12*28×ãF@ ıHÿŞbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28õö=@àHH ãbcluster_0_1/xla_runh*
8
	fusion_37*28Ö¶8@àÌHßíbcluster_0_1/xla_runh
8
	fusion_41*28ˆ8@ÀÎH€àbcluster_0_1/xla_runh
8
	fusion_40*28¼©7@ ©H ÷bcluster_1_1/xla_runh
8
	fusion_33*28œÔ/@à™H€²bcluster_0_1/xla_runh
7
fusion_3*28¹ø)@ îH £bcluster_1_1/xla_runh
8
	fusion_54*28Ûú(@ÀßH  bcluster_0_1/xla_runh
8
	fusion_15*28œ£'@ÀİH€–bcluster_1_1/xla_runh
8
	fusion_16*28ÚÙ @ ©H¿ïbcluster_1_1/xla_runh
E
select_and_scatter_143*28øÿ@€¸H Õbcluster_0_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¼¿@ÀºH Øb4model/dropout_1/dropout/random_uniform/RandomUniformh
7
reduce_1*28İ±@à­H€Çbcluster_0_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾ì@À¤H€Õbtranspose_0h
8
	fusion_17*28ş·@€£Hà³bcluster_1_1/xla_runh
8
	fusion_27*28¾©@ÀŒH ®bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28úé@ ’HÀœb"gradients/transpose_grad/transposeh
6
fusion_4*28ü@à}H€™bcluster_2_1/xla_runh
@
reduce_window_193*28½Æ@€„Hà°bcluster_1_1/xla_runh
§
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ÿ @€H€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
6
reduce_3*28à@€|H ½bcluster_1_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28¿ß@ "H RbCudnnRNNh*
4
reduce*28½¥@ÀvH ‹bcluster_0_1/xla_runh
7
	fusion_56*28§@€jHŸ•bcluster_0_1/xla_runh
6
reduce_2*28½Š@ÿjHà‚bcluster_1_1/xla_runh
7
	fusion_11*28¾à@ÀZH ­bcluster_1_1/xla_runh
6
fusion_8*28½Ç@ÀaHÿ„bcluster_2_1/xla_runh
5
reduce_2*28İ¾@ bHàgbcluster_0_1/xla_runh
5
fusion_5*28Ş@À^HÀgbcluster_1_1/xla_runh
3
fusion*28Ÿ­@àVH ebcluster_8_1/xla_runh
3
fusion*28 ‡@ XHà^bcluster_3_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28½ó@€YH `b4model/dropout_2/dropout/random_uniform/RandomUniformh
5
reduce_4*28¿ï@àRH sbcluster_1_1/xla_runh
5
fusion_2*28ıÁ@¿OHÀvbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28€·@àGHàmb%Adam/Adam/update_12/ResourceApplyAdamh
>
reduce_window_263*28ş”@€THàWbcluster_1_1/xla_runh
6
	fusion_58*28şö@€MH kbcluster_0_1/xla_runh
5
fusion_7*28¼«@àMHÿ_bcluster_0_1/xla_runh
5
reduce_5*28¼¥@ FH€qbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28…@€IHßpb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_24*28 ÷@ JH€Xbcluster_0_1/xla_runh
6
	fusion_43*28Ÿã@ÀJHàQbcluster_0_1/xla_runh
6
	fusion_23*28Ÿµ@ FHàObcluster_2_1/xla_runh
4
fusion*28şÖ@ßCHàLbcluster_10_1/xla_runh
6
	fusion_60*28Í@àCHàJbcluster_0_1/xla_runh
5
fusion_9*28ş–@ ;H€Lbcluster_1_1/xla_runh
3
fusion*28¾é
@€<HàHbcluster_9_1/xla_runh
6
	fusion_10*28 µ
@À1H€qbcluster_1_1/xla_runh
5
fusion_8*28Ş¦
@à<H€Bbcluster_1_1/xla_runh
9
fusion_33__2*28Ÿ…
@¿0HÀHbcluster_2_1/xla_runh
6
	fusion_26*28Şº	@À6H Dbcluster_1_1/xla_runh
6
	fusion_61*28€	@€4Hà=bcluster_0_1/xla_runh
6
	fusion_18*28 ç@À/Hà=bcluster_1_1/xla_runh
4
copy_57*28Şİ@ .Hàabcluster_1_1/xla_runh
4
copy_57*28şÑ@ .H€>bcluster_0_1/xla_runh
6
	fusion_16*28ÿ³@ÿ0HÀ7bcluster_2_1/xla_runh
6
	fusion_30*28¿°@ÿ,H€Rbcluster_2_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ß@À$HÀ=b%Adam/Adam/update_14/ResourceApplyAdamh
4
add_266*28€@ *Hà=bcluster_2_1/xla_runh
6
	fusion_41*28À@ÀHÀ4bcluster_1_1/xla_runh
6
	fusion_43*28€õ@À!HÀ<bcluster_1_1/xla_runh
4
copy_72*28Şñ@à*H >bcluster_0_1/xla_runh
4
copy_50*28ÀÓ@€+H =bcluster_1_1/xla_runh
3
fusion*28ßµ@ß#H€?bcluster_2_1/xla_runh
6
	fusion_42*28ŞŸ@€H <bcluster_1_1/xla_runh
5
fusion_6*28ÿ’@à)H€0bcluster_4_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ÿò@€$HÀ8bcluster_1_1/xla_runh
5
fusion_6*28ßÚ@À#H€6bcluster_2_1/xla_runh
6
	fusion_21*28¿°@À H€.bcluster_2_1/xla_runh
6
	fusion_33*28àš@àH€1bcluster_2_1/xla_runh
6
	fusion_48*28Ş“@  Hà.bcluster_2_1/xla_runh
6
	fusion_45*28¾’@À"Hà.bcluster_1_1/xla_runh
5
fusion_1*28ß…@ !Hà-bcluster_2_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Àø@ "Hà+b
div_no_nanh
6
	fusion_36*28 ó@€ Hà'bcluster_2_1/xla_runh
9
fusion_33__1*28ŸÛ@€Hà%bcluster_2_1/xla_runh
6
	fusion_50*28İÅ@àHà'bcluster_0_1/xla_runh
4
add_368*28 Å@À H€%bcluster_2_1/xla_runh
6
	fusion_49*28¾±@ÀHÀ#bcluster_2_1/xla_runh
6
	fusion_44*28à–@àHà%bcluster_1_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€‘@€HÀ"bAssignAddVariableOp_1h
4
add_343*28û@ŸHà$bcluster_2_1/xla_runh
6
	fusion_42*28¿÷@€Hà(bcluster_2_1/xla_runh
4
add_356*28ßó@€H 'bcluster_2_1/xla_runh
4
add_331*28¾â@ÿH€!bcluster_2_1/xla_runh
3
add_39*28¾Ô@àHŸ!bcluster_4_1/xla_runh
6
	fusion_27*28ÿĞ@ÀHàbcluster_2_1/xla_runh
4
slice_1*28à®@ Hà-bcluster_9_1/xla_runh
3
add_11*28àŸ@ÀHÀbcluster_7_1/xla_runh
3
fusion*28 @€HÀbcluster_7_1/xla_runh
3
fusion*28€—@ÀHÀbcluster_5_1/xla_runh
3
fusion*28à„@ÀHÀbcluster_6_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28àù@€H bAssignAddVariableOp_7h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€"@€"H€"b
LogicalAndh