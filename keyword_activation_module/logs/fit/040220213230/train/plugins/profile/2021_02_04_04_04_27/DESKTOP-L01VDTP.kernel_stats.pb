
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28µ Þ@€«Hºþ0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28«Ø»@¼…!HœØ!bcluster_1_1/xla_runh
F
select_and_scatter_533*28ÌÖ©@œŒ Hüý bcluster_0_1/xla_runh
9
	fusion_15*28’Îú@üùH½½bcluster_0_1/xla_runh
9
	fusion_38*28³•ì@ü‘H¼€bcluster_1_1/xla_runh
9
	fusion_11*28˜…Õ@šHü©bcluster_0_1/xla_runh
8
fusion_6*28¶¯É@ü×Hbcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Àò@´H½’bcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28ñš@€2H€ºXb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28²à“@à)H ¡bCudnnRNNh‰
9
	fusion_25*28Šðü@½ñH¾Äbcluster_1_1/xla_runh
9
	fusion_20*28µ¦ê@ÞæHþábcluster_1_1/xla_runh
9
	fusion_24*28õÿè@½¼HžÌbcluster_1_1/xla_runh
9
	fusion_16*28–ŠË@þ»H¾¶bcluster_0_1/xla_runh
A
reduce_window_119*28Üâ¦@ÿÚHÿ£bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28Å¾þ@ÞáH¿Çbcluster_1_1/xla_runh
Ã
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28™Ëö@àHàcbCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ƒÉä@¿½
HŸ¿bcluster_0_1/xla_runh
9
	fusion_19*28£ÞÛ@Þƒ
HßŠbcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28äŒÐ@€H€jXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_23*28âìÎ@¿À	Hþ©
bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Æ–Ç@ßøHž‹bcluster_0_1/xla_runh
8
reduce_3*28Æ‹Å@¿	HŸå	bcluster_0_1/xla_runh
8
reduce_4*28ˆäÂ@ßƒ	H¿õ	bcluster_0_1/xla_runh
8
fusion_1*28ˆÒÂ@ßîH¿ã	bcluster_1_1/xla_runh
8
reduce_5*28‰È·@ÿÇH¿ˆ	bcluster_0_1/xla_runh
6
reduce*28ÊŒ¶@¿¡H¿¸	bcluster_1_1/xla_runh
8
reduce_1*28‹©´@Ÿ“HŸ…
bcluster_1_1/xla_runh
9
	fusion_39*28ïì•@ßÞHàÌbcluster_1_1/xla_runh
9
	fusion_23*28¯½“@ŸîH¿šbcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ôø|@ÿÈHÿ¯bcluster_1_1/xla_runh
8
	fusion_28*28°š{@À×H€šbcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28“³y@€.HÀ€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28ñÓs@¿¦HŸýbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28—›e@€ÉH¿ÿb2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28ôÎb@Ÿ¾HÀôbcluster_0_1/xla_runh
E
select_and_scatter_313*28—ŸR@€àH€šbcluster_0_1/xla_runh
5
fusion*28úžQ@ÀÙHÀ•bcluster_1_1/xla_runh
8
	fusion_12*28»ÎG@ÀH îbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28¸õ;@€HHÀ¸bcluster_0_1/xla_runh*
8
	fusion_37*28›Ó8@€ÎH€åbcluster_0_1/xla_runh
8
	fusion_41*28Ùÿ7@ßÐHÿßbcluster_0_1/xla_runh
8
	fusion_40*28˜¶6@Ÿ¶Hàòbcluster_1_1/xla_runh
8
	fusion_33*28˜¦1@€ŸH ½bcluster_0_1/xla_runh
7
fusion_3*28ü€*@àåH¿Ÿbcluster_1_1/xla_runh
8
	fusion_15*28 å&@€ÞHÀ”bcluster_1_1/xla_runh
8
	fusion_54*28Üî%@ÀËHÀ÷bcluster_0_1/xla_runh
E
select_and_scatter_143*28›Õ@€±H€Ñbcluster_0_1/xla_runh
7
reduce_1*28Ü¶@ÿ±H Ébcluster_0_1/xla_runh
8
	fusion_16*28ü£@ ¤H Þbcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Ü™@À¶HàÑb4model/dropout_1/dropout/random_uniform/RandomUniformh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28™ü@ÿ¤HÀÕbtranspose_0h
8
	fusion_17*28À¡@à§HÀÄbcluster_1_1/xla_runh
8
	fusion_27*28‹@ ŸH€³bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Õ@Ÿ’Hà™b"gradients/transpose_grad/transposeh
7
fusion_4*28ûŸ@€‚Hßœbcluster_2_1/xla_runh
@
reduce_window_193*28½·@à‡H ›bcluster_1_1/xla_runh
7
reduce_3*28À@ŸH «bcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ø@À|Hà‹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
4
reduce*28Àþ@ gH €bcluster_0_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28ß­@ "Hà[bCudnnRNNh*
6
reduce_2*28˜ë@ iH€„bcluster_1_1/xla_runh
7
	fusion_56*28ý®@ÀgH bcluster_0_1/xla_runh
7
	fusion_11*28þí@ XH —bcluster_1_1/xla_runh
5
fusion_8*28Ÿå@ŸfH€}bcluster_2_1/xla_runh
5
reduce_2*28Ÿ«@€bH fbcluster_0_1/xla_runh
5
fusion_5*28ž@ ^Hàgbcluster_1_1/xla_runh
3
fusion*28€Ê@€[HÀdbcluster_8_1/xla_runh
3
fusion*28¼¡@ÀYHàbbcluster_3_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ÿ“@àXH€ob4model/dropout_2/dropout/random_uniform/RandomUniformh
5
fusion_2*28Üê@€NHàtbcluster_1_1/xla_runh
5
reduce_4*28þÚ@ÀSHàfbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28½Ú@€JH ub%Adam/Adam/update_12/ResourceApplyAdamh
5
reduce_5*28ÿÒ@ GH€rbcluster_1_1/xla_runh
>
reduce_window_263*28¿í@À=HÀ[bcluster_1_1/xla_runh
6
	fusion_58*28¼í@àNHàabcluster_0_1/xla_runh
5
fusion_7*28¾ä@ÀMHÀ]bcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¿¥@ IH€_b%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_23*28¾—@ÀKH€Ubcluster_2_1/xla_runh
6
	fusion_24*28ü‹@¿JHÀWbcluster_0_1/xla_runh
6
	fusion_43*28žˆ@€LHàYbcluster_0_1/xla_runh
4
fusion*28½¶@€CH€Ibcluster_10_1/xla_runh
5
fusion_9*28Ý¢@àAH€Qbcluster_1_1/xla_runh
6
	fusion_60*28ýŸ@€BH¿Gbcluster_0_1/xla_runh
6
	fusion_26*28ü¿
@€7HŸtbcluster_1_1/xla_runh
3
fusion*28½¼
@ÿ<H Bbcluster_9_1/xla_runh
9
fusion_33__2*28 º
@€2HÀIbcluster_2_1/xla_runh
5
fusion_8*28€ÿ	@€;H€Abcluster_1_1/xla_runh
6
	fusion_10*28 Ö	@À0HÀibcluster_1_1/xla_runh
4
copy_57*28ß	@€0H @bcluster_0_1/xla_runh
6
	fusion_18*28Àü@à/H Jbcluster_1_1/xla_runh
4
copy_50*28¾Á@À)Hßkbcluster_1_1/xla_runh
4
add_266*28 ¼@à*H€Cbcluster_2_1/xla_runh
6
	fusion_16*28àº@à1Hà7bcluster_2_1/xla_runh
4
copy_57*28à³@À+Hà:bcluster_1_1/xla_runh
6
	fusion_41*28Àž@À0H€4bcluster_1_1/xla_runh
6
	fusion_61*28þ@à-H€7bcluster_0_1/xla_runh
6
	fusion_30*28ßƒ@À-H 3bcluster_2_1/xla_runh
6
	fusion_43*28¿@€!H€=bcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¿î@€$H ?b%Adam/Adam/update_14/ResourceApplyAdamh
3
fusion*28¿Ò@ß&HÀ<bcluster_2_1/xla_runh
5
fusion_6*28à²@À#H€Abcluster_2_1/xla_runh
6
	fusion_42*28þ@À!H€?bcluster_1_1/xla_runh
4
copy_72*28ƒ@ÿ'H 4bcluster_0_1/xla_runh
5
fusion_6*28î@À'H ,bcluster_4_1/xla_runh
6
	fusion_33*28Þã@ HßGbcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ÞÖ@À#H ?bcluster_1_1/xla_runh
9
fusion_33__1*28 ¾@à!HàSbcluster_2_1/xla_runh
5
fusion_1*28€»@à"Hà/bcluster_2_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à¢@À#HÀ(b
div_no_nanh
6
	fusion_36*28Àõ@ !HÀ'bcluster_2_1/xla_runh
6
	fusion_48*28Ÿî@  H ,bcluster_2_1/xla_runh
6
	fusion_21*28¾æ@  H 'bcluster_2_1/xla_runh
6
	fusion_45*28€Ö@€Hà2bcluster_1_1/xla_runh
6
	fusion_50*28àÒ@ÀH 'bcluster_0_1/xla_runh
4
add_368*28ÀÑ@  H -bcluster_2_1/xla_runh
6
	fusion_49*28¾Ì@€Hà6bcluster_2_1/xla_runh
6
	fusion_42*28 °@ H ,bcluster_2_1/xla_runh
4
add_331*28ß˜@àHà5bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à@ÀH€!bAssignAddVariableOp_7h
3
add_39*28à@ Hà!bcluster_4_1/xla_runh
6
	fusion_44*28ßü@€HÀ&bcluster_1_1/xla_runh
4
add_343*28¿÷@àHà%bcluster_2_1/xla_runh
4
add_356*28Àë@àH ,bcluster_2_1/xla_runh
3
fusion*28€ß@ÀH bcluster_5_1/xla_runh
3
fusion*28ßÕ@ŸHàbcluster_7_1/xla_runh
3
add_11*28àÔ@€Hà!bcluster_7_1/xla_runh
6
	fusion_27*28 Ð@€H  bcluster_2_1/xla_runh
3
fusion*28¾§@àHÀ!bcluster_6_1/xla_runh
4
slice_1*28 ›@€H bcluster_9_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28àˆ@ H bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 "@ "H "b
LogicalAndh