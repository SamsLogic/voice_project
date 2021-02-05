

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ÿ÷¦@¿H¸­Dbcluster_0_1/xla_runh*
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28É‚@ý­H›²0bcluster_0_1/xla_runh*
F
select_and_scatter_533*28èËñ
@ø…BHùåBbcluster_0_1/xla_runh
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28‹²Õ
@¾ØH»’(bcluster_1_1/xla_runh*
9
	fusion_11*28»ÀÂ	@úÖ9Hº«:bcluster_0_1/xla_runh
9
	fusion_15*28ÜË¾	@ú¶9Hùµ:bcluster_0_1/xla_runh
9
	fusion_38*28¾­¹	@Ú’9Hùü9bcluster_1_1/xla_runh
8
fusion_6*28žîµ	@º¡9Hºá9bcluster_0_1/xla_runh
9
	fusion_25*28©ý@œþ$HÛã%bcluster_1_1/xla_runh
9
	fusion_20*28Œ‹ý@ü‹$Hüô$bcluster_1_1/xla_runh
9
	fusion_24*28¬¶ú@œ†$HœØ$bcluster_1_1/xla_runh
:
sgemm_32x32x32_NN_vec*28îÝÁ@ (HÿµbCudnnRNNhê

\
sgemm_32x32x32_NN_vec*28Ñ¯@À3H€”Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophê

£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Õ³¢@ý×H¼Ç%bcluster_0_1/xla_runh
9
	fusion_16*28úüý@Ý“H½îbcluster_0_1/xla_runh
A
reduce_window_119*28¾»Ï@ÜáH½Ãbcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28‚Ì»@±Hübcluster_1_1/xla_runh
9
	fusion_19*28ÔÎ¯@ÞœHžŠbcluster_1_1/xla_runh
8
reduce_4*28´¤@ÞÕHÞ–bcluster_0_1/xla_runh
8
fusion_1*28¶Ì‹@¾ÊHžÚbcluster_1_1/xla_runh
Ã
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ØÇŠ@àHà^bCudnnRNNhÄ
6
reduce*28öµ‚@žïHþŠbcluster_1_1/xla_runh
8
reduce_1*28ÖÏÿ@þíH¾ýbcluster_1_1/xla_runh
8
reduce_3*28÷¶ÿ@ÞôH¾Žbcluster_0_1/xla_runh
8
reduce_5*28Öïý@žþHÞ·bcluster_0_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28“‹û@€HÀjXb(gradients/CudnnRNN_grad/CudnnRNNBackprophÄ
9
	fusion_32*28‚˜«@ÞêH¿Õbcluster_0_1/xla_runh
9
	fusion_39*28æ©@þîHßÆbcluster_1_1/xla_runh
9
	fusion_28*28Âæž@ž½Hžébcluster_0_1/xla_runh
F
select_and_scatter_313*28¿·•@ÞçHþæbcluster_0_1/xla_runh
9
	fusion_23*28‚Ð@ßÖHž¤bcluster_0_1/xla_runh
9
	fusion_23*28¢Ò÷@ß¯Hþ³bcluster_1_1/xla_runh
6
fusion*28Æ¢ó@ÞˆHÿ™bcluster_1_1/xla_runh
[
sgemm_32x32x32_TN_vec*28é‚Ý@À-HÀ„Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph~
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Ë˜Ð@ßâ	Hß‘
b2model/dropout/dropout/random_uniform/RandomUniformh
9
	fusion_37*28ê‰Ã@ßºHÞÚbcluster_0_1/xla_runh
w
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28ÉñÂ@ŸzH¿Úbcluster_0_1/xla_runh*
9
	fusion_47*28îú©@ÿíHßºbcluster_0_1/xla_runh
9
	fusion_17*28°¬¨@ŸàH¿œbcluster_1_1/xla_runh
9
	fusion_12*28Žº¥@ŸÐHÿbcluster_1_1/xla_runh
9
	fusion_33*28ðÙš@àòHÿÔbcluster_0_1/xla_runh
9
	fusion_41*28¯‰˜@ßçHŸíbcluster_0_1/xla_runh
9
	fusion_16*28Ž¡†@ÿ”H¿Übcluster_1_1/xla_runh
7
fusion_3*28ø¡c@ÿ¡H€¤bcluster_1_1/xla_runh
@
reduce_window_193*28—ë_@¿ªH€ûbcluster_1_1/xla_runh
7
fusion_5*28”‚Z@Ÿ÷HÿÄbcluster_1_1/xla_runh
7
reduce_1*28³­S@ßáH¿ bcluster_0_1/xla_runh
8
	fusion_56*28—ÁN@€ÉHà÷bcluster_0_1/xla_runh
8
	fusion_40*28Ú M@ ©HàŽbcluster_1_1/xla_runh
7
reduce_3*28õƒK@€ûHà»bcluster_1_1/xla_runh
5
reduce*28˜¯C@ üH€Çbcluster_0_1/xla_runh
7
reduce_2*28ÙêA@àíHŸ´bcluster_1_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÚÊA@à|H btranspose_0h*
7
reduce_2*28¹¦A@€úHß´bcluster_0_1/xla_runh
8
	fusion_15*28öé@@ÿçHà±bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28›—>@à†H€úb"gradients/transpose_grad/transposeh*
E
select_and_scatter_143*28·™;@ÀØHÿþbcluster_0_1/xla_runh
7
fusion_2*28š¼7@¿¨Hàøbcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ÙÅ4@ ™Hÿßb4model/dropout_1/dropout/random_uniform/RandomUniformh
8
	fusion_27*28º¦4@¿¤H ýbcluster_1_1/xla_runh
8
	fusion_54*28»á2@ ™H Ìbcluster_0_1/xla_runh
8
	fusion_11*28ýà0@€ùH¿Ïbcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¼ë+@ €H Ÿb4model/dropout_2/dropout/random_uniform/RandomUniformh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ºª*@àoHàXb(gradients/CudnnRNN_grad/CudnnRNNBackproph*
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28›ƒ(@À"H ibCudnnRNNhT
7
fusion_9*28Ÿå%@ÀÃHŸbcluster_1_1/xla_runh
8
	fusion_10*28Üô$@ ÍHàóbcluster_1_1/xla_runh
6
copy_50*28ß«"@€µHàïbcluster_1_1/xla_runh
6
copy_57*28½ç @ÿ¶H àbcluster_0_1/xla_runh
8
	fusion_58*28›®@€‚HÀ°bcluster_0_1/xla_runh
6
reduce_5*28ÞÆ@ÀoH –bcluster_1_1/xla_runh
6
fusion_4*28›¥@ ~Hÿ—bcluster_2_1/xla_runh
?
reduce_window_263*28àŒ@À\H€«bcluster_1_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿÛ@ÀHÀŠbtranspose_9h

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28àÖ@àlHà‘b%Adam/Adam/update_12/ResourceApplyAdamh
´
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28½ê@ß|HÀb$gradients/transpose_9_grad/transposeh
6
fusion_8*28Í@ŸcH †bcluster_1_1/xla_runh
6
reduce_4*28þÉ@àqHà€bcluster_1_1/xla_runh
6
fusion_8*28þÄ@ŸaHÀµbcluster_2_1/xla_runh
¬
ƒ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28€Ÿ@ _Hàhbgradients/AddNh
3
fusion*28€œ@ \Hàibcluster_8_1/xla_runh
3
fusion*28€¤@àZHàabcluster_3_1/xla_runh
3
fusion*28ŸŒ@€BH ~bcluster_9_1/xla_runh
6
	fusion_24*28¿Š@€NHÀjbcluster_0_1/xla_runh
6
	fusion_18*28þÁ@€OH gbcluster_1_1/xla_runh
5
fusion_7*28þ”@ÀMHàcbcluster_0_1/xla_runh
4
copy_72*28Àï@ÀLHÀabcluster_0_1/xla_runh
6
	fusion_43*28ÿÔ@¿NHàWbcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Þ³@àKH€cb%Adam/Adam/update_13/ResourceApplyAdamh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28þ•@ EHà_b%Adam/Adam/update_15/ResourceApplyAdamh
4
copy_57*28ÝÌ@€DH Tbcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Ý‰@ÀDHàPb%Adam/Adam/update_16/ResourceApplyAdamh
6
	fusion_41*28ÿÃ@€DH€Lbcluster_1_1/xla_runh
6
	fusion_23*28ÿÀ@à?H Kbcluster_2_1/xla_runh
4
fusion*28À•@À>HÀKbcluster_10_1/xla_runh
6
	fusion_60*28Ÿ÷
@À@HÀEbcluster_0_1/xla_runh
9
fusion_32__2*28žó
@À2HàFbcluster_2_1/xla_runh
4
fusion*28ÿð
@À?HÀFbcluster_11_1/xla_runh
6
	fusion_45*28ßÔ
@À@H€Ebcluster_1_1/xla_runh
6
fusion_1*28½Ë
@À9H Lbcluster_12_1/xla_runh
6
	fusion_44*28Ÿé	@à8Hà=bcluster_1_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ŸÁ	@ %H€Cb%Adam/Adam/update_17/ResourceApplyAdamh
6
	fusion_29*28À®	@À-HÀSbcluster_2_1/xla_runh
6
	fusion_61*28¾ª	@€3HŸHbcluster_0_1/xla_runh
6
	fusion_43*28Þ§	@À4Hà?bcluster_1_1/xla_runh
4
fusion*28Àÿ@ -H€Ebcluster_12_1/xla_runh
6
	fusion_42*28Þý@ 0HÿAbcluster_1_1/xla_runh
6
	fusion_16*28Ÿ½@Ÿ0HÀ6bcluster_2_1/xla_runh
4
add_258*28ž¬@ +HÀ=bcluster_2_1/xla_runh
3
fusion*28à§@ 'Hà<bcluster_2_1/xla_runh
5
fusion_6*28¾@€$Hà=bcluster_2_1/xla_runh
6
	fusion_32*28à©@ )HÀ7bcluster_2_1/xla_runh
6
	fusion_35*28ŸŒ@Ÿ#HÀWbcluster_2_1/xla_runh
6
	fusion_26*28 é@ %Hà8bcluster_1_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28 Â@à#H 7bcluster_1_1/xla_runh
9
fusion_32__1*28ß¬@ $H ,bcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Þ«@À$HŸ-b%Adam/Adam/update_14/ResourceApplyAdamh
5
fusion_6*28 ¨@À%Hà)bcluster_4_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 ›@€#Hà,b
div_no_nanh
4
add_360*28ÿ…@  H€4bcluster_2_1/xla_runh
6
	fusion_47*28ÿ„@€ H 0bcluster_2_1/xla_runh
5
fusion_1*28àé@  H€,bcluster_2_1/xla_runh
6
	fusion_21*28¿é@à HÀ)bcluster_2_1/xla_runh
6
	fusion_50*28ÿÑ@ HÀ&bcluster_0_1/xla_runh
4
add_323*28 Ë@ Hà3bcluster_2_1/xla_runh
6
	fusion_41*28ÿÀ@ H 5bcluster_2_1/xla_runh
6
	fusion_48*28€´@ÀH #bcluster_2_1/xla_runh
4
add_348*28Þ³@àHÀ/bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ÿ@ÀH€$bAssignAddVariableOp_7h
4
add_335*28À÷@ Hà#bcluster_2_1/xla_runh
3
fusion*28 æ@ Hàbcluster_6_1/xla_runh
3
fusion*28€ä@ HÀ$bcluster_7_1/xla_runh
3
fusion*28Àâ@ HÀ"bcluster_5_1/xla_runh
6
	fusion_26*28€á@€H "bcluster_2_1/xla_runh
3
add_11*28€Ü@àH€ bcluster_7_1/xla_runh
5
slice_1*28ßË@¿H€ bcluster_12_1/xla_runh
3
add_39*28ŸË@àHàbcluster_4_1/xla_runh
5
slice_1*28€—@ HÀbcluster_10_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿ€@€H€bAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à @à Hà b
LogicalAndh