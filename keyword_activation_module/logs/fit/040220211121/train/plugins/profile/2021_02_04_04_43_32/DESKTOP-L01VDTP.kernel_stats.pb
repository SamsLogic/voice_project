
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28İõâ@ÿ­HšÂ0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Œ½@Ü!HÜÔ!bcluster_1_1/xla_runh
F
select_and_scatter_533*28®Ì©@œŠ Hüâ bcluster_0_1/xla_runh
9
	fusion_15*28òÑû@‚Hœ¿bcluster_0_1/xla_runh
9
	fusion_38*28Ø‘í@ŸHİ¸bcluster_1_1/xla_runh
9
	fusion_11*28Û”Ô@ü¡HÛbcluster_0_1/xla_runh
8
fusion_6*28¼ğÈ@ıÜHÜûbcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ô‰@ıŠHıÒbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28ñ¸–@à2Hß¸Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28ÕŸ‚@€(H ¥bCudnnRNNh‰
9
	fusion_25*28ññı@İëH½Õbcluster_1_1/xla_runh
9
	fusion_20*28´¾ç@¾éH½İbcluster_1_1/xla_runh
9
	fusion_24*28ö¨â@±H¥bcluster_1_1/xla_runh
9
	fusion_16*28™õÊ@ÉHŞ´bcluster_0_1/xla_runh
A
reduce_window_119*28ÿæ¤@¾ÚHß•bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28„ùş@ßİHß¹bcluster_1_1/xla_runh
Ä
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28‰áî@€HÀ“bCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ˆ®æ@ŸÎ
HßÅbcluster_0_1/xla_runh
²
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28ˆºÜ@ŸH€•Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_19*28†¸Û@ş
HŸİ
bcluster_1_1/xla_runh
9
	fusion_23*28©§Ê@³	H¿Å
bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ÈêÅ@ßúHŞğ
bcluster_0_1/xla_runh
8
reduce_3*28‰èÄ@ß”	Hÿÿ	bcluster_0_1/xla_runh
8
reduce_4*28«ÄÂ@ßÿH¿æ	bcluster_0_1/xla_runh
8
fusion_1*28éØÁ@¿ØHŸ´	bcluster_1_1/xla_runh
8
reduce_1*28ŠÀ¼@ß Hÿ†
bcluster_1_1/xla_runh
6
reduce*28êÀ¹@¿£Hÿ¤	bcluster_1_1/xla_runh
8
reduce_5*28‹½·@¿ÆH¿übcluster_0_1/xla_runh
9
	fusion_39*28®––@ÀßHŸÄbcluster_1_1/xla_runh
9
	fusion_23*28¯‰”@ßöHÿ—bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ôü}@ŸÎHÿ±bcluster_1_1/xla_runh
8
	fusion_28*28¶™{@¿ÑH „bcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28±Áy@ +HÀ€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28·ór@€¢H¿ôbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28•…d@ß¼H¿÷b2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28öâ`@ßÂHÿÖbcluster_0_1/xla_runh
E
select_and_scatter_313*28–Q@ ÚHŸ£bcluster_0_1/xla_runh
5
fusion*28·ŒQ@ÀÙHÿ‚bcluster_1_1/xla_runh
8
	fusion_12*28·H@àÿHßêbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28÷©<@ÀHHÀÊbcluster_0_1/xla_runh*
8
	fusion_37*28»Ö8@àÏH€ïbcluster_0_1/xla_runh
8
	fusion_40*28ø©8@ µH ƒbcluster_1_1/xla_runh
8
	fusion_41*28º8@ ÒH€İbcluster_0_1/xla_runh
8
	fusion_33*28Û›1@ŸH Ğbcluster_0_1/xla_runh
7
fusion_3*28üŠ*@¿åHÀ™bcluster_1_1/xla_runh
8
	fusion_54*28»æ)@ÀéHÀ™bcluster_0_1/xla_runh
8
	fusion_15*28û´&@€ÙH šbcluster_1_1/xla_runh
E
select_and_scatter_143*28Ü‘ @àºH€×bcluster_0_1/xla_runh
8
	fusion_16*28û…@à¦H€Óbcluster_1_1/xla_runh
7
reduce_1*28¿@€°H€Çbcluster_0_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28½Õ@À§H Æbtranspose_0h
8
	fusion_17*28Şù@à¢Hà·bcluster_1_1/xla_runh
8
	fusion_27*28Ûë@ÿ‹H€®bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28½Ò@À“Hÿšb"gradients/transpose_grad/transposeh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28İä@ HÀ£b4model/dropout_2/dropout/random_uniform/RandomUniformh
?
reduce_window_193*28›Ï@€xHßôbcluster_1_1/xla_runh
6
fusion_4*28›Æ@ zHŸšbcluster_2_1/xla_runh
7
reduce_3*28Ÿ‰@ HÀ–bcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28Şé@à{H€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
4
reduce*28ÛÀ@ŸtH€„bcluster_0_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28¾ı@À"H€PbCudnnRNNh*
7
	fusion_56*28½Ó@€kHßšbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Ÿ¢@ sHàwb4model/dropout_1/dropout/random_uniform/RandomUniformh
5
reduce_2*28ıÇ@€lH xbcluster_1_1/xla_runh
5
fusion_8*28›°@€gHàsbcluster_2_1/xla_runh
7
	fusion_11*28Ş›@àYH€†bcluster_1_1/xla_runh
5
reduce_2*28¹@ÀbH€gbcluster_0_1/xla_runh
5
fusion_5*28ş@àNHàkbcluster_1_1/xla_runh
5
reduce_4*28àŞ@€XH ibcluster_1_1/xla_runh
3
fusion*28 @ XHŸkbcluster_8_1/xla_runh
3
fusion*28ŸŒ@ YHà_bcluster_3_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28»Ç@€HHŸob%Adam/Adam/update_12/ResourceApplyAdamh
5
fusion_2*28àø@€NHÀrbcluster_1_1/xla_runh
>
reduce_window_263*28ßô@à>HàXbcluster_1_1/xla_runh
6
	fusion_58*28ıÌ@ÀKH bbcluster_0_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28İÅ@àHH€fb%Adam/Adam/update_13/ResourceApplyAdamh
5
fusion_7*28¾Ÿ@ÀLH€[bcluster_0_1/xla_runh
6
	fusion_23*28¾ü@ÀJH€Tbcluster_2_1/xla_runh
6
	fusion_24*28şû@àKHàZbcluster_0_1/xla_runh
6
	fusion_43*28Ûî@ÿJHÀSbcluster_0_1/xla_runh
5
reduce_5*28¾Í@€FHÿobcluster_1_1/xla_runh
4
fusion*28¼Ë@ŸDH€Mbcluster_10_1/xla_runh
5
fusion_9*28«@àAHÀPbcluster_1_1/xla_runh
6
	fusion_60*28¤@€AHàGbcluster_0_1/xla_runh
3
fusion*28ıë
@€>H Fbcluster_9_1/xla_runh
9
fusion_33__2*28İ”
@ 3HßGbcluster_2_1/xla_runh
5
fusion_8*28ÿâ	@À:H =bcluster_1_1/xla_runh
6
	fusion_26*28ÀŸ	@€(H Abcluster_1_1/xla_runh
6
	fusion_10*28àı@€0H vbcluster_1_1/xla_runh
6
	fusion_18*28ÿÏ@ )H Cbcluster_1_1/xla_runh
4
copy_57*28ıÎ@¿.Hÿ;bcluster_0_1/xla_runh
6
	fusion_16*28½Ç@à1H€;bcluster_2_1/xla_runh
6
	fusion_41*28¾¶@À/H€Dbcluster_1_1/xla_runh
6
	fusion_61*28€—@À/H 8bcluster_0_1/xla_runh
4
copy_50*28Ş@à+H `bcluster_1_1/xla_runh
6
	fusion_30*28ß@ /H€5bcluster_2_1/xla_runh
4
copy_72*28€@ *HÀLbcluster_0_1/xla_runh
4
copy_57*28ßõ@À,HÀ:bcluster_1_1/xla_runh
4
add_266*28ŸË@À-HÀ0bcluster_2_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ÀÅ@ $H€>b%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_43*28à¼@à#HàUbcluster_1_1/xla_runh
5
fusion_6*28€«@ *H /bcluster_4_1/xla_runh
3
fusion*28ßñ@à&HÀ2bcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ŞÅ@€$H€7bcluster_1_1/xla_runh
5
fusion_1*28ŸÁ@€#H€.bcluster_2_1/xla_runh
5
fusion_6*28À¦@à#H 6bcluster_2_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 ¦@à#Hà,b
div_no_nanh
6
	fusion_48*28 „@À Hà(bcluster_2_1/xla_runh
6
	fusion_36*28¿ù@€ Hà'bcluster_2_1/xla_runh
6
	fusion_33*28à÷@ÀH€7bcluster_2_1/xla_runh
6
	fusion_42*28€ó@€!HÀ1bcluster_1_1/xla_runh
9
fusion_33__1*28Àñ@ÀH Abcluster_2_1/xla_runh
6
	fusion_21*28¿í@  H€(bcluster_2_1/xla_runh
6
	fusion_50*28ÀÇ@€Hà%bcluster_0_1/xla_runh
4
add_368*28ß»@ÀH $bcluster_2_1/xla_runh
6
	fusion_44*28 µ@ÀH€'bcluster_1_1/xla_runh
6
	fusion_49*28Ÿ«@àH€$bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à @€HÀ"bAssignAddVariableOp_7h
6
	fusion_45*28À @ Hà"bcluster_1_1/xla_runh
6
	fusion_42*28 …@ÀH€*bcluster_2_1/xla_runh
4
add_343*28€î@ H  bcluster_2_1/xla_runh
6
	fusion_27*28Àá@€H€&bcluster_2_1/xla_runh
3
add_39*28á@ŸH¿"bcluster_4_1/xla_runh
4
add_331*28ÿŞ@àH€bcluster_2_1/xla_runh
3
add_11*28İ@€H bcluster_7_1/xla_runh
4
slice_1*28ŸÛ@ÀH $bcluster_9_1/xla_runh
3
fusion*28 Ù@ÀHÀ bcluster_5_1/xla_runh
3
fusion*28ÀØ@ÀH€bcluster_7_1/xla_runh
4
add_356*28 Ì@àH€bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ş¢@ÿHÀ3bAssignAddVariableOp_1h
3
fusion*28¾–@ H bcluster_6_1/xla_runh
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à!@à!Hà!b
LogicalAndh