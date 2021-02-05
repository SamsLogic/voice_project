
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28¸©ß@ÿ«HÚ˜0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28ì“º@œ†!HÜ¾!bcluster_1_1/xla_runh
F
select_and_scatter_533*28ÎÂª@¼Ÿ Hœß bcluster_0_1/xla_runh
9
	fusion_15*28²àú@ı„Hı²bcluster_0_1/xla_runh
9
	fusion_38*28Ñ¡ê@ÜHœìbcluster_1_1/xla_runh
9
	fusion_11*28¶ôÓ@ı˜HœÎbcluster_0_1/xla_runh
8
fusion_6*28ù»Ì@İâHİ¸bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28á²‹@ı™H½èbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28Ïò£@ 3HÀ¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28á–‘@ (HÀ¥bCudnnRNNh‰
9
	fusion_25*28‘Øÿ@şáH¾Øbcluster_1_1/xla_runh
9
	fusion_20*28Õƒè@ŞÙHìbcluster_1_1/xla_runh
9
	fusion_24*28ôòç@¾áH¾Ìbcluster_1_1/xla_runh
9
	fusion_16*28™øÈ@¾ÈHşóbcluster_0_1/xla_runh
A
reduce_window_119*28û¨¥@ŞĞHŞ bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28âùı@ÿÜHş²bcluster_1_1/xla_runh
Ã
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28Û¼ô@àH ZbCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28„šå@ÿº
H¸bcluster_0_1/xla_runh
9
	fusion_19*28ÈÑÚ@Ÿ‘
H¾ã
bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28§îÓ@€H€iXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_23*28†êË@ß±	Hÿª
bcluster_1_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ªèÅ@¿	H¾ˆbcluster_0_1/xla_runh
8
reduce_3*28‡ÁÅ@Ÿ–	H¿‹
bcluster_0_1/xla_runh
8
reduce_4*28æ¬Ã@¿†	H¿Ò	bcluster_0_1/xla_runh
8
fusion_1*28ÉÀ@ÿòH¿À	bcluster_1_1/xla_runh
8
reduce_1*28è×¹@ŸœHŸ
bcluster_1_1/xla_runh
8
reduce_5*28Êö¸@ŸÏH¿‚	bcluster_0_1/xla_runh
6
reduce*28êÔµ@ÿ‘Hÿ­	bcluster_1_1/xla_runh
9
	fusion_39*28Íü–@ßİHŸÌbcluster_1_1/xla_runh
9
	fusion_23*28­á“@ŸôHÿ’bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28‘£@¿ÚH¿¹bcluster_1_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28´Õy@€.Hà‡Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_28*28Ò¡y@¿ËH¿ûbcluster_0_1/xla_runh
8
	fusion_47*28°˜t@¿H¿ïbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28²ñd@ŸÄHàùb2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28’ìa@àÉHŸäbcluster_0_1/xla_runh
E
select_and_scatter_313*28™R@ÀĞH€†bcluster_0_1/xla_runh
5
fusion*28–øQ@€ŞHÿ™bcluster_1_1/xla_runh
8
	fusion_12*28ºıF@ÀƒH€íbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28ÜÌ<@€HHŸËbcluster_0_1/xla_runh*
8
	fusion_37*28ÙÁ8@ ÍHÿãbcluster_0_1/xla_runh
8
	fusion_41*28Ú8@ŸÑH€Şbcluster_0_1/xla_runh
8
	fusion_40*28˜Ò7@à»Hßÿbcluster_1_1/xla_runh
8
	fusion_33*28™Ì0@ šHÿÁbcluster_0_1/xla_runh
7
fusion_3*28İˆ,@àéH€Ïbcluster_1_1/xla_runh
8
	fusion_54*28¼›(@ ÚH ‰bcluster_0_1/xla_runh
8
	fusion_15*28ü´'@ áH€bcluster_1_1/xla_runh
E
select_and_scatter_143*28ÛÅ!@à¾HàÙbcluster_0_1/xla_runh
8
	fusion_16*28ü@À©H¿ábcluster_1_1/xla_runh
7
reduce_1*28¹‚@€­H¿Ébcluster_0_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28®@€¨H€Ğbtranspose_0h
8
	fusion_17*28›ê@ ŸHÿÀbcluster_1_1/xla_runh
8
	fusion_27*28»…@à‰Hà²bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28şï@€“Hàb"gradients/transpose_grad/transposeh
6
reduce_3*28»¦@ÀHßÁbcluster_1_1/xla_runh
?
reduce_window_193*28Şß@¿yH «bcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ûÌ@ ‹H€£b4model/dropout_2/dropout/random_uniform/RandomUniformh
6
fusion_4*28İ¨@àzH ›bcluster_2_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ı‡@àHàXb(gradients/CudnnRNN_grad/CudnnRNNBackproph
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28¾¾@€#H€TbCudnnRNNh*
3
reduce*28ßê@ÀqH€~bcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28İœ@ sH€wb4model/dropout_1/dropout/random_uniform/RandomUniformh
7
	fusion_11*28şÈ@ÀYHŸbcluster_1_1/xla_runh
6
	fusion_56*28½˜@ÀdHà|bcluster_0_1/xla_runh
5
reduce_2*28€ï@ hH tbcluster_1_1/xla_runh
6
fusion_8*28şè@ÀgH€‚bcluster_2_1/xla_runh
5
reduce_2*28ÀÈ@ cH gbcluster_0_1/xla_runh
5
fusion_5*28¿¼@ ]Hà{bcluster_1_1/xla_runh
3
fusion*28Ÿ–@€WH abcluster_8_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ÿƒ@ JH ub%Adam/Adam/update_12/ResourceApplyAdamh
3
fusion*28Şƒ@€XH _bcluster_3_1/xla_runh
5
reduce_4*28ßü@ PHÀ{bcluster_1_1/xla_runh
5
reduce_5*28½ù@ÀGHàrbcluster_1_1/xla_runh
5
fusion_2*28ß¢@ÀNH€abcluster_1_1/xla_runh
>
reduce_window_263*28Şğ@à:HàWbcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ŸÈ@ÀIHÀwb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_58*28ÀÂ@ MHàabcluster_0_1/xla_runh
5
fusion_7*28¿¥@ÀLHàXbcluster_0_1/xla_runh
6
	fusion_24*28¿@àJH€Wbcluster_0_1/xla_runh
6
	fusion_43*28Ÿ€@ KH Sbcluster_0_1/xla_runh
6
	fusion_23*28ŞÅ@ IHàQbcluster_2_1/xla_runh
6
	fusion_60*28 Î@àBH€Kbcluster_0_1/xla_runh
4
fusion*28ß¼@€BH€Hbcluster_10_1/xla_runh
5
fusion_9*28ß¡@ŸAH€Jbcluster_1_1/xla_runh
3
fusion*28½@À>HàLbcluster_9_1/xla_runh
9
fusion_33__2*28ş©
@à2HÀJbcluster_2_1/xla_runh
6
	fusion_10*28€…
@ 0H€jbcluster_1_1/xla_runh
5
fusion_8*28À„
@À:Hà@bcluster_1_1/xla_runh
6
	fusion_26*28ÿú	@ß6H€Hbcluster_1_1/xla_runh
4
copy_50*28€×	@ ,Hàcbcluster_1_1/xla_runh
4
copy_57*28€	@À/H€?bcluster_0_1/xla_runh
6
	fusion_18*28¿ô@¿/H Bbcluster_1_1/xla_runh
6
	fusion_61*28ŸŞ@¿3Hà7bcluster_0_1/xla_runh
6
	fusion_16*28ÀÙ@€0HÀLbcluster_2_1/xla_runh
4
copy_57*28¿Ã@à+HàHbcluster_1_1/xla_runh
6
	fusion_41*28Ÿ @À1H€3bcluster_1_1/xla_runh
6
	fusion_30*28À™@ ,H 6bcluster_2_1/xla_runh
4
add_266*28Ş—@À,H¿<bcluster_2_1/xla_runh
6
	fusion_42*28ÿô@À!H Obcluster_1_1/xla_runh
6
	fusion_43*28Àí@ $Hà9bcluster_1_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ÀÎ@€%Hà<b%Adam/Adam/update_14/ResourceApplyAdamh
4
copy_72*28ßÁ@à(HàLbcluster_0_1/xla_runh
5
fusion_6*28Ÿ–@À)H€.bcluster_4_1/xla_runh
3
fusion*28Ş“@à%Hà:bcluster_2_1/xla_runh
5
fusion_6*28 à@€!Hà9bcluster_2_1/xla_runh
5
fusion_1*28€Ô@à#Hà0bcluster_2_1/xla_runh
6
	fusion_33*28€Ì@€ H Sbcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28³@ !H :bcluster_1_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¾@ÿ"H -b
div_no_nanh
6
	fusion_21*28€›@À HÀ.bcluster_2_1/xla_runh
6
	fusion_50*28Ÿ“@À Hà,bcluster_0_1/xla_runh
6
	fusion_45*28ÿ€@ H .bcluster_1_1/xla_runh
6
	fusion_36*28ßÿ@€ H€,bcluster_2_1/xla_runh
9
fusion_33__1*28€ó@ÀH€-bcluster_2_1/xla_runh
6
	fusion_48*28 ç@€ H€&bcluster_2_1/xla_runh
6
	fusion_49*28àÆ@ÀH€$bcluster_2_1/xla_runh
4
add_368*28ŸÆ@À HÀ&bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28Ÿ‰@€H $bAssignAddVariableOp_7h
3
add_39*28¿ÿ@¿Hà bcluster_4_1/xla_runh
6
	fusion_27*28ßù@ÀH "bcluster_2_1/xla_runh
6
	fusion_44*28€÷@ Hà&bcluster_1_1/xla_runh
6
	fusion_42*28€ò@ÀH€*bcluster_2_1/xla_runh
4
add_331*28€ï@ÀHà&bcluster_2_1/xla_runh
4
add_343*28€í@àHÀ&bcluster_2_1/xla_runh
4
slice_1*28ßß@ÀH€!bcluster_9_1/xla_runh
4
add_356*28ÀÆ@àHàbcluster_2_1/xla_runh
3
fusion*28à@ÀHÀbcluster_7_1/xla_runh
3
fusion*28ş˜@àHÀbcluster_5_1/xla_runh
3
add_11*28ÿ•@ÀHÀbcluster_7_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿ…@ÿH€bAssignAddVariableOp_1h
3
fusion*28ßƒ@€H bcluster_6_1/xla_runh
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€ @€ H€ b
LogicalAndh