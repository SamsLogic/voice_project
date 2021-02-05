
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Ø˜á@  HÚ–0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28éÉ»@œ!H¼Á!bcluster_1_1/xla_runh
F
select_and_scatter_533*28Ì¿©@œ Hœ‡!bcluster_0_1/xla_runh
9
	fusion_15*28‘óú@¼†Hœµbcluster_0_1/xla_runh
9
	fusion_38*28¶Âë@İ›H‡bcluster_1_1/xla_runh
9
	fusion_11*28¸ÛÓ@ H¾bcluster_0_1/xla_runh
8
fusion_6*28½ĞË@ı×Hœ’bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ŞÊ™@İÎH½›bcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28íØ±@à1H€»Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28ÏÓ•@€*HŸ·bCudnnRNNh‰
9
	fusion_25*28Òãÿ@¾ïH¾Êbcluster_1_1/xla_runh
9
	fusion_20*28ôé@¾ôHşŞbcluster_1_1/xla_runh
9
	fusion_24*28“†è@æHİÜbcluster_1_1/xla_runh
9
	fusion_16*28—èÉ@ŞºHş€bcluster_0_1/xla_runh
A
reduce_window_119*28¼Â¥@ßÃH¾¾bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28Ìş@ÿïHŸ³bcluster_1_1/xla_runh
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Á–ö@¾Ã
H¾Öbcluster_0_1/xla_runh
Ã
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ƒœò@€H xbCudnnRNNhá
²
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28‹·è@ HÀŠXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_19*28¥‰ß@¿Œ
HŞş
bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28çòÏ@ß”	H¾¼bcluster_0_1/xla_runh
9
	fusion_23*28¨¢Ï@¾°	Hÿ¾
bcluster_1_1/xla_runh
8
reduce_3*28§¨Ã@¿‡	H¿Ú	bcluster_0_1/xla_runh
8
reduce_4*28ˆãÁ@ŸôH¿¼	bcluster_0_1/xla_runh
8
fusion_1*28ˆÁ@ŸİH¾Õ	bcluster_1_1/xla_runh
8
reduce_5*28ÉÛ¹@ßÊHßÑ	bcluster_0_1/xla_runh
6
reduce*28Šë´@¿HÖ	bcluster_1_1/xla_runh
8
reduce_1*28¨ü³@Ÿ“H¾Ì	bcluster_1_1/xla_runh
9
	fusion_39*28í±•@ŸØHßºbcluster_1_1/xla_runh
9
	fusion_23*28ˆ•@¿ıHà›bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28@¿ÈHŸ¾bcluster_1_1/xla_runh
8
	fusion_28*28ğ{@ÿßHß’bcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28³Ğx@€,HÀşXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28’¥q@€™H€Çbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28±°b@¿ªHßûb2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28²¾a@ÀÀHÿäbcluster_0_1/xla_runh
E
select_and_scatter_313*28÷ïS@ßçH bcluster_0_1/xla_runh
5
fusion*28Õ‹Q@ÿĞHÿ›bcluster_1_1/xla_runh
8
	fusion_12*28Ø–H@¿“H¿ÿbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28Ôú>@ HHÀÔbcluster_0_1/xla_runh*
8
	fusion_37*28ù8@ ÍHŸæbcluster_0_1/xla_runh
8
	fusion_41*28¹…8@ÿÏHàábcluster_0_1/xla_runh
8
	fusion_40*28½6@à¤Hàâbcluster_1_1/xla_runh
8
	fusion_33*28ùÉ1@ß›HŸÖbcluster_0_1/xla_runh
7
fusion_3*28ù§-@àöH ·bcluster_1_1/xla_runh
8
	fusion_54*28ú´)@ èHà“bcluster_0_1/xla_runh
8
	fusion_15*28ºÊ'@ÀÚHÀbcluster_1_1/xla_runh
E
select_and_scatter_143*28Üà @ßºHÀ×bcluster_0_1/xla_runh
8
	fusion_16*28ıá@à¥Hà‰bcluster_1_1/xla_runh
7
reduce_1*28½œ@À®H Îbcluster_0_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28™ê@¿—H¿Òb4model/dropout_1/dropout/random_uniform/RandomUniformh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾Ø@À«H¿Îbtranspose_0h
8
	fusion_17*28ÿ@à£HŸµbcluster_1_1/xla_runh
8
	fusion_27*28¾¿@à‰Hàµbcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ş‰@€•H œb"gradients/transpose_grad/transposeh
?
reduce_window_193*28Ÿ­@ HÀÖbcluster_1_1/xla_runh
6
fusion_4*28Ã@ yH€–bcluster_2_1/xla_runh
6
reduce_3*28½±@À}H Ébcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ø™@ÿ{Hà‹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28İè@À"Hà^bCudnnRNNh*
4
reduce*28Şˆ@àpHà„bcluster_0_1/xla_runh
6
reduce_2*28¼“@ànH bcluster_1_1/xla_runh
7
	fusion_56*28¾÷@àgH€bcluster_0_1/xla_runh
7
	fusion_11*28ßã@à_H€´bcluster_1_1/xla_runh
6
fusion_8*28ß¸@àeHÀ³bcluster_2_1/xla_runh
5
reduce_2*28Ÿ¹@àbH fbcluster_0_1/xla_runh
5
fusion_5*28Üˆ@ÿ]HÀhbcluster_1_1/xla_runh
3
fusion*28ı§@àWH€cbcluster_8_1/xla_runh
3
fusion*28ş¤@€ZH€bbcluster_3_1/xla_runh
6
	fusion_58*28Ş–@ÀNH kbcluster_0_1/xla_runh
5
reduce_4*28Ÿ’@ÀPH zbcluster_1_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ıº@€VH€Zb4model/dropout_2/dropout/random_uniform/RandomUniformh
>
reduce_window_263*28Ù@¿UHÿYbcluster_1_1/xla_runh
5
fusion_2*28‚@ OHàrbcluster_1_1/xla_runh
5
fusion_7*28ÿâ@€MH€dbcluster_0_1/xla_runh
6
	fusion_24*28ÿ£@àKHà[bcluster_0_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28À’@ IH€gb%Adam/Adam/update_12/ResourceApplyAdamh
6
	fusion_23*28ÿ@ LHàVbcluster_2_1/xla_runh
5
reduce_5*28Şú@€FHàpbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾ñ@ŸHH Xb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_43*28àï@àJHàRbcluster_0_1/xla_runh
4
fusion*28¼Ğ@ AHÀRbcluster_10_1/xla_runh
5
fusion_9*28“@€<HÀMbcluster_1_1/xla_runh
6
	fusion_60*28Ş€@ @H€Gbcluster_0_1/xla_runh
3
fusion*28€ê
@À=H Gbcluster_9_1/xla_runh
9
fusion_33__2*28ÿ±
@à<HÀFbcluster_2_1/xla_runh
5
fusion_8*28…
@€;HÀAbcluster_1_1/xla_runh
6
	fusion_10*28¾ş	@ 1H qbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28½è	@À%HÀCb%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_26*28 ß	@ 7H Lbcluster_1_1/xla_runh
4
copy_57*28½Ó	@à/H Bbcluster_0_1/xla_runh
6
	fusion_18*28Ÿ…	@À2HÀRbcluster_1_1/xla_runh
6
	fusion_61*28ßô@ 3HÀFbcluster_0_1/xla_runh
4
add_266*28Ÿ»@à*HŸFbcluster_2_1/xla_runh
6
	fusion_16*28Û¸@ 0H€9bcluster_2_1/xla_runh
6
	fusion_43*28 ­@€,H€Lbcluster_1_1/xla_runh
6
	fusion_30*28¾¥@ .H€8bcluster_2_1/xla_runh
6
	fusion_41*28À¢@ 1H€5bcluster_1_1/xla_runh
4
copy_57*28ÿ¡@€,HÀHbcluster_1_1/xla_runh
4
copy_50*28Àƒ@€,HÀHbcluster_1_1/xla_runh
6
	fusion_42*28 ğ@À H€?bcluster_1_1/xla_runh
3
fusion*28ßÀ@À'H€:bcluster_2_1/xla_runh
4
copy_72*28à±@à(HÀ=bcluster_0_1/xla_runh
5
fusion_6*28¿ú@€#H€Zbcluster_2_1/xla_runh
5
fusion_6*28Àô@À(H .bcluster_4_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28€Ú@ $Hà=bcluster_1_1/xla_runh
6
	fusion_48*28¾Å@à Hÿ0bcluster_2_1/xla_runh
6
	fusion_33*28ŸÁ@ !H€6bcluster_2_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À–@à#H€+b
div_no_nanh
6
	fusion_45*28Ş@€H€7bcluster_1_1/xla_runh
6
	fusion_21*28¿†@àHÀ,bcluster_2_1/xla_runh
6
	fusion_49*28¿…@ !H -bcluster_2_1/xla_runh
6
	fusion_50*28ßü@ÀHÀ.bcluster_0_1/xla_runh
9
fusion_33__1*28ÿù@À!H€,bcluster_2_1/xla_runh
5
fusion_1*28àë@àHà&bcluster_2_1/xla_runh
6
	fusion_36*28ÀÙ@ÀH€2bcluster_2_1/xla_runh
6
	fusion_44*28ÿÌ@ HÀ&bcluster_1_1/xla_runh
4
add_368*28àÂ@À H€$bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28½Ÿ@ßHÀ#bAssignAddVariableOp_7h
4
add_331*28Ÿù@àH€%bcluster_2_1/xla_runh
6
	fusion_27*28Ÿà@ÀHàbcluster_2_1/xla_runh
6
	fusion_42*28şÜ@ÿH€-bcluster_2_1/xla_runh
3
add_39*28ÀÜ@àH€"bcluster_4_1/xla_runh
4
add_356*28¼Ù@€H€ bcluster_2_1/xla_runh
3
fusion*28ÀÔ@€Hàbcluster_5_1/xla_runh
4
add_343*28€Ô@àH #bcluster_2_1/xla_runh
3
add_11*28ÀÒ@ H€bcluster_7_1/xla_runh
3
fusion*28ÿÉ@ÀHÀbcluster_7_1/xla_runh
4
slice_1*28À«@€Hà!bcluster_9_1/xla_runh
3
fusion*28À¥@ H€bcluster_6_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ı…@€H bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À!@À!HÀ!b
LogicalAndh