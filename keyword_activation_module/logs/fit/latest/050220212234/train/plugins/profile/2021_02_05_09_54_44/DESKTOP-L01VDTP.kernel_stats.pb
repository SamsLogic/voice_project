

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28±¡¶@üïHÙ³Dbcluster_0_1/xla_runh*
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28İ“@ÜºHºÂ0bcluster_0_1/xla_runh*
F
select_and_scatter_533*28¨ï
@øåAHùÎBbcluster_0_1/xla_runh
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28äÍÕ
@ŞH¼(bcluster_1_1/xla_runh*
9
	fusion_11*28´•Â	@Ùé9H™¢:bcluster_0_1/xla_runh
9
	fusion_15*28’¿	@º¶9H¹ :bcluster_0_1/xla_runh
9
	fusion_38*28“±º	@Ú®9H™€:bcluster_1_1/xla_runh
8
fusion_6*28“³¶	@šš9H™Ú9bcluster_0_1/xla_runh
9
	fusion_25*28£ˆ‘@Üş$Hüé%bcluster_1_1/xla_runh
\
sgemm_32x32x32_NN_vec*28ÇŠŒ@ 3HÀºXb(gradients/CudnnRNN_grad/CudnnRNNBackprophê

9
	fusion_20*28èü@Üã#Hœ…%bcluster_1_1/xla_runh
9
	fusion_24*28éÃù@œë#H¼Õ$bcluster_1_1/xla_runh
:
sgemm_32x32x32_NN_vec*28Ô™Ë@€(HßµbCudnnRNNhê

£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Ò@½ÖHüë$bcluster_0_1/xla_runh
9
	fusion_16*28º•ş@¼—Hİèbcluster_0_1/xla_runh
A
reduce_window_119*28şÖÍ@İÑHİÉbcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28½”¾@ÊHüùbcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28œŒÒ@€H€qXb(gradients/CudnnRNN_grad/CudnnRNNBackprophÄ
9
	fusion_19*28ÏÚ¯@ı Hş÷bcluster_1_1/xla_runh
Ã
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28î˜¤@ßHàabCudnnRNNhÄ
8
reduce_4*28’“Œ@şÎHşƒbcluster_0_1/xla_runh
8
fusion_1*28´ûŠ@¾¾H¾Ÿbcluster_1_1/xla_runh
6
reduce*28Ôı@¾õHöbcluster_1_1/xla_runh
8
reduce_1*28•¦€@¾ùH¾âbcluster_1_1/xla_runh
8
reduce_5*28‘ùş@öHşµbcluster_0_1/xla_runh
8
reduce_3*28”¯ü@îH¼bcluster_0_1/xla_runh
9
	fusion_32*28 ˜«@şìHÿŞbcluster_0_1/xla_runh
9
	fusion_39*28ÿà¨@ÿäHŞ¹bcluster_1_1/xla_runh
9
	fusion_28*28ı­Ÿ@¾ÇHßæbcluster_0_1/xla_runh
F
select_and_scatter_313*28Ûş•@¾şHÿÎbcluster_0_1/xla_runh
9
	fusion_23*28Üˆ@¿ÜHübcluster_0_1/xla_runh
9
	fusion_23*28àÏö@Ş³HŞ›bcluster_1_1/xla_runh
6
fusion*28ÂÆñ@ŸˆHß’bcluster_1_1/xla_runh
[
sgemm_32x32x32_TN_vec*28æ¤ß@À,HàˆXb(gradients/CudnnRNN_grad/CudnnRNNBackproph~
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ŠÖÔ@ŸÏH¿Ñ
b2model/dropout/dropout/random_uniform/RandomUniformh
9
	fusion_37*28éÎÊ@¿¶H¿Ìbcluster_0_1/xla_runh
w
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28‰ÙÂ@À{HŸäbcluster_0_1/xla_runh*
9
	fusion_47*28¬Ù«@ÿÿH¿ïbcluster_0_1/xla_runh
9
	fusion_17*28ø§@ßäHŸ¡bcluster_1_1/xla_runh
9
	fusion_12*28îÂ¤@Ÿ¸HŸ¢bcluster_1_1/xla_runh
9
	fusion_41*28®†™@ßéHß–bcluster_0_1/xla_runh
9
	fusion_33*28…™@ÿıHŸ¼bcluster_0_1/xla_runh
9
	fusion_16*28ñÇ…@Ÿ„H¿Übcluster_1_1/xla_runh
7
fusion_3*28×¼b@ß¬H¿ùbcluster_1_1/xla_runh
@
reduce_window_193*28ôë`@à¨Hßîbcluster_1_1/xla_runh
7
fusion_5*28—ÏX@ ğH Óbcluster_1_1/xla_runh
7
reduce_1*28øŞR@ŸãH€Ğbcluster_0_1/xla_runh
8
	fusion_56*28²‰O@ŸĞH ûbcluster_0_1/xla_runh
8
	fusion_40*28ÚÁL@À H€„bcluster_1_1/xla_runh
7
reduce_3*28»ØF@€şH€Äbcluster_1_1/xla_runh
7
reduce_2*28¸–C@àæH€ºbcluster_1_1/xla_runh
7
reduce_2*28ÔB@ ôH Óbcluster_0_1/xla_runh
5
reduce*28™§B@¿óHß´bcluster_0_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28·ÜA@À|H ¢btranspose_0h*
8
	fusion_15*28›Ù@@ßùHß¤bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28›Ú=@À‡Hàõb"gradients/transpose_grad/transposeh*
E
select_and_scatter_143*28úÓ:@ ÒH¿€bcluster_0_1/xla_runh
7
fusion_2*28Ú±7@ÀœH öbcluster_1_1/xla_runh
8
	fusion_27*28ûÎ3@À¡H€ıbcluster_1_1/xla_runh
8
	fusion_11*28Ùı0@àôHŸÍbcluster_1_1/xla_runh
8
	fusion_54*28Ûì/@À‰H¿Ãbcluster_0_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28›â+@ÀêH¿­b4model/dropout_1/dropout/random_uniform/RandomUniformh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ıĞ)@ÀrHÀXb(gradients/CudnnRNN_grad/CudnnRNNBackproph*
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28Øè'@ "HàlbCudnnRNNhT
7
fusion_9*28Æ&@€ÌHàŠbcluster_1_1/xla_runh
6
copy_50*28ü¨#@ßÃHà†bcluster_1_1/xla_runh
8
	fusion_10*28ü¹"@ ¨HÀîbcluster_1_1/xla_runh
6
copy_57*28è @€ºH€ßbcluster_0_1/xla_runh
8
	fusion_58*28½Á@¿H ©bcluster_0_1/xla_runh
6
reduce_5*28œŒ@ÀoH€”bcluster_1_1/xla_runh
6
fusion_4*28İ@ßwH –bcluster_2_1/xla_runh
?
reduce_window_263*28İÙ@àoH ¡bcluster_1_1/xla_runh
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Şû@ qH  b4model/dropout_2/dropout/random_uniform/RandomUniformh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾¾@ Hà†btranspose_9h
6
fusion_8*28ş¦@€fH §bcluster_1_1/xla_runh
´
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¹â@ÿ|H b$gradients/transpose_9_grad/transposeh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾§@ÀnHÀ•b%Adam/Adam/update_12/ResourceApplyAdamh
6
reduce_4*28Ÿ£@ oH €bcluster_1_1/xla_runh
6
fusion_8*28¦@€]Hàbcluster_2_1/xla_runh
3
fusion*28ÿÚ@À[H€mbcluster_8_1/xla_runh
¬
ƒ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28Ÿ§@€ZHàgbgradients/AddNh
6
	fusion_18*28Àı@€RHàtbcluster_1_1/xla_runh
3
fusion*28@€XH¿_bcluster_3_1/xla_runh
4
fusion*28ßƒ@€AH ‡bcluster_9_1/xla_runh
6
	fusion_24*28€°@€MHÀhbcluster_0_1/xla_runh
5
fusion_7*28ÿÃ@€MH cbcluster_0_1/xla_runh
6
	fusion_43*28½´@€NHÿUbcluster_0_1/xla_runh
4
copy_57*28€ô@€CH Sbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28úó@ÀIH Tb%Adam/Adam/update_13/ResourceApplyAdamh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¿è@àHH Sb%Adam/Adam/update_15/ResourceApplyAdamh
4
copy_72*28¿Ñ@ÀHH€Rbcluster_0_1/xla_runh
4
fusion*28¾ä@À?H Ubcluster_10_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ŞÈ@ŸBHàJb%Adam/Adam/update_16/ResourceApplyAdamh
6
	fusion_41*28ŸÅ@€EHàJbcluster_1_1/xla_runh
6
	fusion_60*28Şš@ß@H Fbcluster_0_1/xla_runh
6
fusion_1*28À™@à3H Lbcluster_12_1/xla_runh
6
	fusion_23*28@ @H Hbcluster_2_1/xla_runh
9
fusion_32__2*28àæ
@À<HÀNbcluster_2_1/xla_runh
4
fusion*28ş×
@à>H¿Dbcluster_11_1/xla_runh
6
	fusion_45*28¿Ò
@ @H Dbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28Àú	@ :Hà?b%Adam/Adam/update_17/ResourceApplyAdamh
6
	fusion_44*28¾ä	@ 9H <bcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ßÑ	@à0H€?b%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_29*28 ÿ@à-H Wbcluster_2_1/xla_runh
6
	fusion_43*28ßì@à2Hà:bcluster_1_1/xla_runh
4
fusion*28ŸÌ@€0HßEbcluster_12_1/xla_runh
6
	fusion_16*28ÿ½@À1Hà7bcluster_2_1/xla_runh
6
	fusion_42*28¾·@ ,HÀ=bcluster_1_1/xla_runh
6
	fusion_61*28€¡@À(HÀ?bcluster_0_1/xla_runh
4
add_258*28À”@à)Hà;bcluster_2_1/xla_runh
3
fusion*28şû@à(HÀ?bcluster_2_1/xla_runh
5
fusion_6*28€µ@À#Hà9bcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ßå@ #Hà9bcluster_1_1/xla_runh
6
	fusion_32*28 Ä@À!H€1bcluster_2_1/xla_runh
5
fusion_6*28À·@ &H€+bcluster_4_1/xla_runh
5
fusion_1*28 ¥@À"Hà,bcluster_2_1/xla_runh
6
	fusion_50*28£@ $Hà)bcluster_0_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ß¢@à#Hà*b
div_no_nanh
6
	fusion_35*28€Š@à H ,bcluster_2_1/xla_runh
6
	fusion_47*28Ÿ„@  Hà-bcluster_2_1/xla_runh
9
fusion_32__1*28À€@ !H *bcluster_2_1/xla_runh
6
	fusion_26*28ş@à!Hÿ/bcluster_1_1/xla_runh
4
add_360*28¿â@àH >bcluster_2_1/xla_runh
6
	fusion_21*28 İ@À H -bcluster_2_1/xla_runh
6
	fusion_48*28¾¯@ HÀ$bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28  @àH€&bAssignAddVariableOp_1h
3
add_39*28¿Œ@ H !bcluster_4_1/xla_runh
6
	fusion_41*28ß‡@ÀHà$bcluster_2_1/xla_runh
4
add_323*28à‚@àHÀ(bcluster_2_1/xla_runh
4
add_348*28¿ş@ÀH€$bcluster_2_1/xla_runh
4
add_335*28ßø@àHà#bcluster_2_1/xla_runh
6
	fusion_26*28¿æ@€HŸ#bcluster_2_1/xla_runh
5
slice_1*28ŞĞ@ŸH€bcluster_12_1/xla_runh
5
slice_1*28 ¾@àH bcluster_10_1/xla_runh
3
fusion*28àª@€H€ bcluster_7_1/xla_runh
3
fusion*28ß¨@àHÀbcluster_6_1/xla_runh
3
add_11*28 ¤@àHà!bcluster_7_1/xla_runh
3
fusion*28€Ÿ@ÀH€bcluster_5_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ß‡@ÿH€bAssignAddVariableOp_7h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿ@ÿHÿb
LogicalAndh