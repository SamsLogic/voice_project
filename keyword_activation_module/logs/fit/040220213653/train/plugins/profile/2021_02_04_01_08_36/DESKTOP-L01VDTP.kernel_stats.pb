
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28•ÙÙ@ß¢HºŠ0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28¬ˆº@Üƒ!HÜº!bcluster_1_1/xla_runh
F
select_and_scatter_533*28ÍŞ©@½› Hüà bcluster_0_1/xla_runh
9
	fusion_15*28ÒÌû@ıúHıËbcluster_0_1/xla_runh
9
	fusion_38*28´ëè@İHœİbcluster_1_1/xla_runh
9
	fusion_11*28•ÏÕ@İšHü”bcluster_0_1/xla_runh
8
fusion_6*28ù¹Ë@½áHİ£bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28€Ê‰@ˆHİÍbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28°¯Œ@À2H€¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28Î¤‹@ (H€£bCudnnRNNh‰
9
	fusion_25*28“ƒ@şüHŞÈbcluster_1_1/xla_runh
9
	fusion_20*28”Ùê@¾HŞìbcluster_1_1/xla_runh
9
	fusion_24*28Óòå@şÇHşÃbcluster_1_1/xla_runh
9
	fusion_16*28˜´È@Ÿ½H÷bcluster_0_1/xla_runh
A
reduce_window_119*28øä¥@şÕHŞ¸bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28ºÚş@ßãHşÃbcluster_1_1/xla_runh
Ã
_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28£©î@àHàWbCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28…áê@ßÃ
HşÂbcluster_0_1/xla_runh
9
	fusion_19*28Éğİ@ß
Hÿü
bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28£¬Ğ@€HàjXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_23*28èØÏ@ß½	HÿÏ
bcluster_1_1/xla_runh
8
reduce_3*28ÇºÄ@ÿ™	HÒ	bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ŠõÃ@ßóHß¶
bcluster_0_1/xla_runh
8
reduce_4*28‰ˆÃ@ß‡	HßÅ	bcluster_0_1/xla_runh
8
fusion_1*28‰˜Â@ÿãH¿Ô	bcluster_1_1/xla_runh
8
reduce_5*28§˜»@ÿËH¾Ğ	bcluster_0_1/xla_runh
6
reduce*28Êı·@ÿ‘Hßê	bcluster_1_1/xla_runh
8
reduce_1*28‹¢³@ÿ›Hßù	bcluster_1_1/xla_runh
9
	fusion_39*28‹á—@ßğHßÂbcluster_1_1/xla_runh
9
	fusion_23*28í×”@ÿıHŸœbcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ò”~@ÀâHŸµbcluster_1_1/xla_runh
8
	fusion_28*28Òx@àÅHŸğbcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28’ıw@€-Hà„Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28•t@€¢Hÿôbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28³òd@ÿÊHß‚b2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28±Ía@ß¿Hàôbcluster_0_1/xla_runh
E
select_and_scatter_313*28¹ÎQ@ŸÚHÿŠbcluster_0_1/xla_runh
5
fusion*28Ò†P@ßËH „bcluster_1_1/xla_runh
8
	fusion_12*28ÛšH@ÿ“H àbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28šÿ<@àGHÀÕbcluster_0_1/xla_runh*
8
	fusion_37*28µ»8@¿ÌHÀêbcluster_0_1/xla_runh
8
	fusion_41*28Ú¥8@àĞHàébcluster_0_1/xla_runh
8
	fusion_40*28øä7@ÀH ÷bcluster_1_1/xla_runh
8
	fusion_33*28Ø…0@à•H¿·bcluster_0_1/xla_runh
7
fusion_3*28úŠ,@€åHÀÍbcluster_1_1/xla_runh
8
	fusion_54*28‚(@ÀÙH †bcluster_0_1/xla_runh
8
	fusion_15*28›í&@àÕHÿ„bcluster_1_1/xla_runh
E
select_and_scatter_143*28ù³!@À¼H€Ùbcluster_0_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Ş·@€¹HàÑb4model/dropout_1/dropout/random_uniform/RandomUniformh
8
	fusion_16*28»£@à§HÀ÷bcluster_1_1/xla_runh
7
reduce_1*28Ÿ‰@À¬H€Ébcluster_0_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ş@€¢HÿÀbtranspose_0h
8
	fusion_17*28ÿº@À¢Hàµbcluster_1_1/xla_runh
8
	fusion_27*28½‡@€H ²bcluster_1_1/xla_runh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ŞÈ@€“HÀb"gradients/transpose_grad/transposeh
6
fusion_4*28ıÙ@à{H ˜bcluster_2_1/xla_runh
6
reduce_3*28 @ {H »bcluster_1_1/xla_runh
?
reduce_window_193*28ş™@ sHà£bcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28ş‰@¿Hà‹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
4
reduce*28ş°@àtH€ƒbcluster_0_1/xla_runh
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28¾ó@ "HÀSbCudnnRNNh*
5
reduce_2*28Àò@ÀlHà~bcluster_1_1/xla_runh
7
	fusion_11*28½İ@ YH¿”bcluster_1_1/xla_runh
7
	fusion_56*28½Ë@ÀeH€bcluster_0_1/xla_runh
5
fusion_8*28À@€aH zbcluster_2_1/xla_runh
5
reduce_2*28ÜÒ@ÀcHŸhbcluster_0_1/xla_runh
5
fusion_5*28@à]HÀibcluster_1_1/xla_runh
3
fusion*28Ÿ°@€YH cbcluster_8_1/xla_runh
3
fusion*28¾ˆ@ XHà`bcluster_3_1/xla_runh
5
fusion_2*28àê@ÀOH€vbcluster_1_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¼é@àXHÀ`b4model/dropout_2/dropout/random_uniform/RandomUniformh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ıİ@àIHàkb%Adam/Adam/update_12/ResourceApplyAdamh
5
reduce_4*28»Ù@¿PH fbcluster_1_1/xla_runh
>
reduce_window_263*28½¢@àSH Ybcluster_1_1/xla_runh
6
	fusion_58*28¿ø@ÀKHÀfbcluster_0_1/xla_runh
5
fusion_7*28ŞÅ@ÀMHàXbcluster_0_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28½¥@àGH€ab%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_24*28Ÿ@ÀJHßXbcluster_0_1/xla_runh
6
	fusion_23*28Ş‰@ÀKHÿSbcluster_2_1/xla_runh
5
reduce_5*28¾ÿ@ FH qbcluster_1_1/xla_runh
6
	fusion_43*28Ûù@àJHàUbcluster_0_1/xla_runh
5
fusion_9*28ŸÈ@à;H€Nbcluster_1_1/xla_runh
4
fusion*28¿¿@àCH€Ibcluster_10_1/xla_runh
3
fusion*28€¢@À>HÀNbcluster_9_1/xla_runh
6
	fusion_60*28¾Ÿ@À@H€Ibcluster_0_1/xla_runh
9
fusion_33__2*28¾Í
@à<HÀJbcluster_2_1/xla_runh
5
fusion_8*28¿Ÿ
@À<HÀAbcluster_1_1/xla_runh
6
	fusion_26*28´	@ 5H¿Hbcluster_1_1/xla_runh
6
	fusion_10*28Ş£	@ 1H€_bcluster_1_1/xla_runh
4
copy_57*28¿›	@ 2HÀ<bcluster_0_1/xla_runh
6
	fusion_18*28ÀÆ@À2Hà:bcluster_1_1/xla_runh
6
	fusion_16*28İÅ@À0HŸ7bcluster_2_1/xla_runh
6
	fusion_61*28ÿÄ@À/H 9bcluster_0_1/xla_runh
4
copy_50*28Ş¶@€*H€bbcluster_1_1/xla_runh
4
copy_57*28 @€,Hàgbcluster_1_1/xla_runh
6
	fusion_41*28¾œ@À/H€6bcluster_1_1/xla_runh
6
	fusion_43*28À’@€&H <bcluster_1_1/xla_runh
6
	fusion_30*28À„@€-Hà4bcluster_2_1/xla_runh
6
	fusion_42*28ıû@À"Hà?bcluster_1_1/xla_runh
4
add_266*28Ÿæ@ +H€Cbcluster_2_1/xla_runh

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28À°@ %H 9b%Adam/Adam/update_14/ResourceApplyAdamh
4
copy_72*28€¥@À)H 2bcluster_0_1/xla_runh
3
fusion*28Àû@€"Hà5bcluster_2_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28€È@À!H€7bcluster_1_1/xla_runh
5
fusion_6*28À¿@à"H 5bcluster_2_1/xla_runh
5
fusion_6*28Ÿ·@ &H€*bcluster_4_1/xla_runh
6
	fusion_33*28à¡@€!H 1bcluster_2_1/xla_runh
6
	fusion_48*28€š@€!H ,bcluster_2_1/xla_runh
5
fusion_1*28¿ù@€!Hß'bcluster_2_1/xla_runh
Ã
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à÷@ "H€*b
div_no_nanh
6
	fusion_49*28ÿô@ !H -bcluster_2_1/xla_runh
9
fusion_33__1*28 ğ@àHÀ,bcluster_2_1/xla_runh
6
	fusion_45*28Ÿì@àH 6bcluster_1_1/xla_runh
6
	fusion_50*28 å@ H€0bcluster_0_1/xla_runh
6
	fusion_21*28€å@  H 'bcluster_2_1/xla_runh
4
add_368*28 Æ@€!H $bcluster_2_1/xla_runh
6
	fusion_36*28Àº@ HÀ%bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28½’@ßHà!bAssignAddVariableOp_7h
6
	fusion_44*28 ı@€H€%bcluster_1_1/xla_runh
6
	fusion_42*28Ş÷@¿H€(bcluster_2_1/xla_runh
4
add_331*28 ÷@€H€+bcluster_2_1/xla_runh
4
slice_1*28ŸÕ@€Hß bcluster_9_1/xla_runh
3
add_39*28àÔ@ H€bcluster_4_1/xla_runh
6
	fusion_27*28€Ô@ Hàbcluster_2_1/xla_runh
4
add_343*28€Ğ@ HÀ bcluster_2_1/xla_runh
4
add_356*28àÌ@ÀHÀbcluster_2_1/xla_runh
3
fusion*28À©@€Hàbcluster_7_1/xla_runh
3
fusion*28¾›@àHÀbcluster_5_1/xla_runh
3
add_11*28€˜@àH bcluster_7_1/xla_runh
3
fusion*28À…@àHÀbcluster_6_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿ‚@€HàbAssignAddVariableOp_1h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à @à Hà b
LogicalAndh