

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28²²°@¼úHØ˜Ebcluster_0_1/xla_runh*
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28Û¿@İ«HúÉ0bcluster_0_1/xla_runh*
F
select_and_scatter_691*28â ï
@¸ãAH¹ÖBbcluster_0_1/xla_runh
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28†ÌÏ
@ıÁH¼°(bcluster_1_1/xla_runh*
9
	fusion_20*28·åÃ	@úß9Hù¦:bcluster_0_1/xla_runh
9
	fusion_24*28÷“½	@Ú¾9Hº’:bcluster_0_1/xla_runh
9
	fusion_56*28ÔĞº	@¹9HÚ›:bcluster_1_1/xla_runh
9
	fusion_15*28Öó´	@™¡9HÙÄ9bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28¨Ì@ºÃ/Hºå2bcluster_0_1/xla_runh
9
	fusion_25*28œëø@ÛÆ)HûÑ+bcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28§ßš@à1HŸÃXb(gradients/CudnnRNN_grad/CudnnRNNBackprophê

9
	fusion_31*28æÔ@¼ğ$H¼È%bcluster_1_1/xla_runh
9
	fusion_26*28¨ñù@üÄ#HœÕ$bcluster_1_1/xla_runh
9
	fusion_30*28Ê¶ö@¼¶#HÜŞ$bcluster_1_1/xla_runh
:
sgemm_32x32x32_NN_vec*28²ßÊ@€)H ¸bCudnnRNNhê

£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ş®æ@½ÊHİæbcluster_0_1/xla_runh
A
reduce_window_180*28›õÌ@½İH¼bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28şƒ½@£H½ğbcluster_1_1/xla_runh
9
	fusion_35*28ˆüƒ@ı°H½âbcluster_0_1/xla_runh
²
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28®Şå@€H€¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophÄ
9
	fusion_25*28¯Ì¬@Ş€Hébcluster_1_1/xla_runh
Ä
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ô°˜@ÀH ˆbCudnnRNNhÄ
7
copy_66*28’¡@´H¾²bcluster_1_1/xla_runh
9
	reduce_15*28²“@şÙH¾®bcluster_0_1/xla_runh
8
fusion_1*28õÉŠ@¾¯HşŒbcluster_1_1/xla_runh
8
reduce_2*28õĞ‚@şèHŞ¾bcluster_1_1/xla_runh
8
reduce_3*28ôµ‚@ŞãHŞ›bcluster_1_1/xla_runh
8
fusion_8*28öüı@ŞùHŞªbcluster_0_1/xla_runh
9
	reduce_16*28µ¼ı@ŞúHş­bcluster_0_1/xla_runh
9
	reduce_14*28µŠü@¾ñHŞÙbcluster_0_1/xla_runh
9
	fusion_22*28Ö¤ë@øHŞÏbcluster_1_1/xla_runh
9
	fusion_24*28º®ç@Ÿ°HŞÔbcluster_1_1/xla_runh
9
	fusion_33*28·Ãä@¾HŞªbcluster_1_1/xla_runh
w
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28Õƒà@àyHßÉbcluster_0_1/xla_runh?
9
	fusion_51*28Ÿß«@ŸëHÿåbcluster_0_1/xla_runh
9
	fusion_47*28¾¥Ÿ@Ş¾HŸóbcluster_0_1/xla_runh
F
select_and_scatter_442*28è”@ßãH¿Íbcluster_0_1/xla_runh
9
	fusion_42*28Àƒ@ßŞHşbcluster_0_1/xla_runh
9
	fusion_57*28º‰@ß¬HŸ‚bcluster_1_1/xla_runh
è
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28€ü@ÿŞ	HÆb6model_2/dropout_6/dropout/random_uniform/RandomUniformh
9
	fusion_29*28ˆ½÷@Ÿ¨HŞ¿bcluster_1_1/xla_runh
6
fusion*28…šõ@Ş«Hş‘bcluster_1_1/xla_runh
[
sgemm_32x32x32_TN_vec*28äãŞ@€,Hß…Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph~
9
	fusion_52*28çˆÈ@ß 	Hÿã	bcluster_0_1/xla_runh
9
	fusion_15*28‹•Æ@ÿœ	HŸè	bcluster_1_1/xla_runh
8
reduce_4*28Ìß¿@ ÄHŸñ	bcluster_1_1/xla_runh
8
reduce_5*28©õ»@¿àHŞ´	bcluster_1_1/xla_runh
9
	fusion_66*28« »@ß¿HŸÃ	bcluster_0_1/xla_runh
9
	fusion_36*28¬ª®@ÿêHÿİbcluster_1_1/xla_runh
9
	fusion_76*28ÌÖ«@ŸİHÿ½bcluster_0_1/xla_runh
9
	fusion_20*28­²¨@ŸÙHÀ¢bcluster_1_1/xla_runh
9
	fusion_38*28®Õ@ÿ¦Hÿ¤bcluster_1_1/xla_runh
9
	fusion_70*28í—“@¿ÚHŸ¢bcluster_0_1/xla_runh
9
	fusion_62*28ò¦@ŸHÿğbcluster_0_1/xla_runh
9
	fusion_19*28‘Íƒ@ŸêH¿ábcluster_1_1/xla_runh
7
fusion_3*28”£f@ÿ¹Hÿ®bcluster_1_1/xla_runh
6
copy_58*28³Ğb@Ÿ´Hÿ‹bcluster_1_1/xla_runh
@
reduce_window_303*28óÒ_@ÿ—HŸíbcluster_1_1/xla_runh
7
fusion_5*28¸®W@àäHàÈbcluster_1_1/xla_runh
8
	reduce_11*28Ø’S@ÿäH€œbcluster_0_1/xla_runh
8
	fusion_85*28·›N@¿ÈH€şbcluster_0_1/xla_runh
8
	fusion_58*28¸¨K@€Hßıbcluster_1_1/xla_runh
7
reduce_7*28—†E@¿äHàbcluster_1_1/xla_runh
7
reduce_6*28º¨C@¿íHÀÍbcluster_1_1/xla_runh
8
	reduce_10*28¹ˆC@¿úH Ïbcluster_0_1/xla_runh
ç
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ÙúA@À°Hßİb6model_2/dropout_7/dropout/random_uniform/RandomUniformh
8
	fusion_18*28ùğA@ÀşH¿ªbcluster_1_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28šÚ@@€}HŸšbtranspose_0h*
8
	reduce_12*28Ü´@@ŸöHà©bcluster_0_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28™Å=@ †H€ôb"gradients/transpose_grad/transposeh*
E
select_and_scatter_193*28˜â:@ ÔHŸbcluster_0_1/xla_runh
8
	fusion_39*28ùì5@€ŸHŸbcluster_1_1/xla_runh
7
fusion_2*28µ4@àƒHß‚bcluster_1_1/xla_runh
8
	fusion_83*28—Û0@¿ˆH Ìbcluster_0_1/xla_runh
8
	fusion_14*28ø«0@àäH€Òbcluster_1_1/xla_runh
ç
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28š“*@À‰H ½b6model_2/dropout_8/dropout/random_uniform/RandomUniformh
7
fusion_9*28Û…(@ ºHßŸbcluster_1_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28»ÿ'@ "Hß}bCudnnRNNhT
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28›ş'@ jH ‰Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph*
6
copy_57*28ÿÖ#@€ËH übcluster_1_1/xla_runh
8
	fusion_10*28ÜÈ"@ ½H¿êbcluster_1_1/xla_runh
6
copy_64*28İ’"@ »H Şbcluster_0_1/xla_runh
8
	fusion_13*28»À@À¥H ébcluster_1_1/xla_runh
8
	fusion_11*28»¡@à©HàŞbcluster_1_1/xla_runh
7
	fusion_88*28²@àH€œbcluster_0_1/xla_runh
6
reduce_9*28Ÿ@ zHàÊbcluster_1_1/xla_runh
?
reduce_window_422*28ûØ@ _HÀ’bcluster_1_1/xla_runh
7
	reduce_11*28ıÔ@ßmHÀ“bcluster_1_1/xla_runh
6
fusion_4*28Ş’@À{H ˜bcluster_2_1/xla_runh
›
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ßÌ@àH •btranspose_9h
´
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿé@ß|HÀb$gradients/transpose_9_grad/transposeh
6
fusion_8*28€À@àiHÀ—bcluster_1_1/xla_runh
6
reduce_8*28¾Ù@ iHÿ„bcluster_1_1/xla_runh
7
	reduce_10*28 Æ@àoHÀŠbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28úç@ßnHÀwb%Adam/Adam/update_18/ResourceApplyAdamh
6
fusion_8*28üÂ@àaH€bcluster_2_1/xla_runh
4
fusion*28ıÏ@¿UHÀ‡bcluster_9_1/xla_runh
3
fusion*28şÁ@ß]HÀmbcluster_8_1/xla_runh
¬
ƒ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28¾‘@ÿZHÀhbgradients/AddNh
6
	fusion_21*28ÿÎ@ RHßpbcluster_1_1/xla_runh
4
copy_81*28ş @ PHàlbcluster_0_1/xla_runh
3
fusion*28¿@ XHÀ`bcluster_3_1/xla_runh
6
	fusion_53*28ÿ»@ÀNH€bbcluster_0_1/xla_runh
6
	fusion_16*28ß²@ OH `bcluster_0_1/xla_runh
6
	fusion_43*28ı«@ÀNH€ibcluster_0_1/xla_runh
6
	fusion_26*28¼Ã@ LH¿gbcluster_0_1/xla_runh
6
	fusion_72*28À•@ MHàUbcluster_0_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28şá@àHH Rb%Adam/Adam/update_19/ResourceApplyAdamh
4
copy_64*28şŞ@ CHÀVbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 ¶@ HH€Qb%Adam/Adam/update_21/ResourceApplyAdamh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾²@ DH€Tb%Adam/Adam/update_22/ResourceApplyAdamh
6
fusion_1*28ç@À=Hßbbcluster_12_1/xla_runh
6
	fusion_23*28ÿÕ@€CHßJbcluster_2_1/xla_runh
6
	fusion_59*28àÏ@ÀEH Kbcluster_1_1/xla_runh
6
	fusion_64*28Ş¯@€AH€Gbcluster_1_1/xla_runh
5
fusion_5*28ÿ@à6HàNbcluster_0_1/xla_runh
4
fusion*28À@€?HÀKbcluster_10_1/xla_runh
4
fusion*28€â
@ @H Jbcluster_11_1/xla_runh
6
	fusion_91*28İİ
@€?H Dbcluster_0_1/xla_runh
9
fusion_32__2*28¿Ù
@À=HÀGbcluster_2_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28 º
@À<H€Db%Adam/Adam/update_23/ResourceApplyAdamh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28•
@À;HàAb%Adam/Adam/update_20/ResourceApplyAdamh
6
	fusion_92*28ş	@€,H Cbcluster_0_1/xla_runh
6
	fusion_16*28¾Æ@À0H€:bcluster_2_1/xla_runh
4
fusion*28 °@À.H <bcluster_12_1/xla_runh
4
add_258*28ß¢@à)H€Gbcluster_2_1/xla_runh
6
	fusion_29*28à‚@ +HÀCbcluster_2_1/xla_runh
3
fusion*28¾ä@ 'H >bcluster_2_1/xla_runh
6
	fusion_61*28ß­@€$Hà7bcluster_1_1/xla_runh
6
	fusion_13*28  @€*HÀ.bcluster_4_1/xla_runh
6
	fusion_60*28à—@€"H€Gbcluster_1_1/xla_runh
6
	fusion_62*28à@€'H€6bcluster_1_1/xla_runh
5
fusion_6*28Ÿ†@€#H 5bcluster_2_1/xla_runh
6
	fusion_34*28€ò@ "Hàebcluster_1_1/xla_runh
6
	fusion_37*28¿Ã@€H 3bcluster_1_1/xla_runh
9
fusion_32__1*28¿¹@À!H abcluster_2_1/xla_runh
6
	fusion_32*28ß¸@€!Hà0bcluster_2_1/xla_runh
6
	fusion_63*28€°@ "HÀ-bcluster_1_1/xla_runh
4
copy_50*28ÿ¥@à#H€+bcluster_1_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28¿Ÿ@À#Hà6bcluster_1_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ #H /b
div_no_nanh
6
	fusion_32*28à—@€#H€3bcluster_1_1/xla_runh
4
copy_98*28ß@à"H€-bcluster_0_1/xla_runh
6
	fusion_21*28ßŒ@À H€.bcluster_2_1/xla_runh
6
	fusion_35*28À‹@à!HÀ+bcluster_2_1/xla_runh
6
	fusion_47*28€†@À H€,bcluster_2_1/xla_runh
6
	fusion_79*28€…@ "H *bcluster_0_1/xla_runh
5
fusion_1*28 ÷@  Hà-bcluster_2_1/xla_runh
6
	fusion_12*28ÀŞ@ H (bcluster_4_1/xla_runh
4
add_360*28ÿ»@  H #bcluster_2_1/xla_runh
6
	fusion_48*28à®@ÀH€#bcluster_2_1/xla_runh
6
	fusion_65*28 ©@ÀH $bcluster_1_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾¡@ H€%bAssignAddVariableOp_1h
4
add_323*28ßı@€HÀ(bcluster_2_1/xla_runh
4
add_348*28¿û@àHÀ$bcluster_2_1/xla_runh
4
add_335*28Àõ@€Hà$bcluster_2_1/xla_runh
6
	fusion_41*28€í@àHà$bcluster_2_1/xla_runh
6
	fusion_26*28Àã@ÀH 'bcluster_2_1/xla_runh
3
add_57*28€Ö@àHàbcluster_4_1/xla_runh
5
slice_1*28€Î@€Hà%bcluster_12_1/xla_runh
5
slice_1*28ÀÌ@€Hà bcluster_10_1/xla_runh
3
fusion*28À®@àH€"bcluster_6_1/xla_runh
3
fusion*28À¨@àH !bcluster_5_1/xla_runh
3
fusion*28À¨@àHà bcluster_7_1/xla_runh
3
add_11*28 §@€HÀ bcluster_7_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿ@ HÀ#bAssignAddVariableOp_7h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à@àHàb
LogicalAndh