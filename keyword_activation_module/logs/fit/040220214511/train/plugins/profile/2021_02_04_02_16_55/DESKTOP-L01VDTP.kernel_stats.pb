
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ØÞ@¿­Hú¹0bcluster_0_1/xla_runh*
®
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28¬‡»@¼Œ!HœÂ!bcluster_1_1/xla_runh
F
select_and_scatter_533*28Íˆª@üŒ Hüò bcluster_0_1/xla_runh
9
	fusion_15*28±ïú@½„H¼±bcluster_0_1/xla_runh
9
	fusion_38*28±Æé@Ü…HœÒbcluster_1_1/xla_runh
9
	fusion_11*28ÕÛÓ@½—H¼Âbcluster_0_1/xla_runh
8
fusion_6*28ºòÉ@ÝäHÝ€bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28áš‰@Ý¢HÝåbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28ò§ž@ 2H ¹Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph‰
:
sgemm_32x32x32_NN_vec*28ì†’@À)Hà¤bCudnnRNNh‰
9
	fusion_25*28ðÖþ@ÞêHž½bcluster_1_1/xla_runh
9
	fusion_20*28´âé@¾ÏHÝÏbcluster_1_1/xla_runh
9
	fusion_24*28òâæ@ÞÀHýÈbcluster_1_1/xla_runh
9
	fusion_16*28˜ÓÈ@þÂHþïbcluster_0_1/xla_runh
A
reduce_window_119*28ÚÍ£@¾ÖH¾¨bcluster_1_1/xla_runh
¤
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28À¿ý@ŸæH¾±bcluster_1_1/xla_runh
Ã
ž_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28©žô@àHàabCudnnRNNhá
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28âîå@ž¶
HÞ¶bcluster_0_1/xla_runh
9
	fusion_19*28§áÜ@ŸŒ
H¿‹bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28á÷Ú@àH¿lXb(gradients/CudnnRNN_grad/CudnnRNNBackprophá
9
	fusion_23*28§þÍ@ß¾	HŸµ
bcluster_1_1/xla_runh
8
reduce_3*28ª§Æ@ß—	Hÿë	bcluster_0_1/xla_runh

m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28ˆšÅ@ßùHÞŽbcluster_0_1/xla_runh
8
reduce_4*28¨§Ã@ÿøHßÚ	bcluster_0_1/xla_runh
8
fusion_1*28‰ÓÂ@ŸàH¿á	bcluster_1_1/xla_runh
8
reduce_5*28É»·@ÿÅH¿‡	bcluster_0_1/xla_runh
6
reduce*28©Š·@ÿHÿÇ	bcluster_1_1/xla_runh
8
reduce_1*28ÊÙµ@ÿŒHþŒ
bcluster_1_1/xla_runh
9
	fusion_39*28î„•@ŸâHß»bcluster_1_1/xla_runh
9
	fusion_23*28Ï×“@¿ïHŸ˜bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28×ƒ|@ ÌH ™bcluster_1_1/xla_runh
8
	fusion_28*28¯¤y@¿ÆH¿ƒbcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28´£x@ .HàÿXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28ñ„s@ “HŸßbcluster_0_1/xla_runh
ã
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ÔÝg@€ÙH€–b2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28·â`@ ¹Hàábcluster_0_1/xla_runh
E
select_and_scatter_313*28µþP@€×H€‹bcluster_0_1/xla_runh
5
fusion*28öðP@ÀÎHÀbcluster_1_1/xla_runh
8
	fusion_12*28ù¿F@ÀHàÔbcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28›Î=@ßGHà½bcluster_0_1/xla_runh*
8
	fusion_37*28™™8@€ËH ábcluster_0_1/xla_runh
8
	fusion_41*28Úî7@€ÑH¿Úbcluster_0_1/xla_runh
8
	fusion_40*28×Æ7@¿¼Hàãbcluster_1_1/xla_runh
8
	fusion_33*28™»0@àHßÃbcluster_0_1/xla_runh
7
fusion_3*28Þ”+@àèHÀ¤bcluster_1_1/xla_runh
8
	fusion_54*28ºÑ(@àÛH Ÿbcluster_0_1/xla_runh
E
select_and_scatter_143*28Ù„"@€ÄH¿Übcluster_0_1/xla_runh
7
reduce_1*28š@ ´HÀÇbcluster_0_1/xla_runh
8
	fusion_16*28¼‹@ ¡HŸébcluster_1_1/xla_runh
œ
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28žÇ@ £HàÉbtranspose_0h
8
	fusion_27*28Û›@À HŸÉbcluster_1_1/xla_runh
8
	fusion_17*28ß€@àŸHÀ¶bcluster_1_1/xla_runh
å
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Þ±@ÀHàìb4model/dropout_2/dropout/random_uniform/RandomUniformh
³
ô_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¯@€‘H ˜b"gradients/transpose_grad/transposeh
ä
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28Üà@€wHß¤b4model/dropout_1/dropout/random_uniform/RandomUniformh
6
reduce_3*28ºÔ@ H¿¿bcluster_1_1/xla_runh
6
fusion_4*28½Ñ@ }H –bcluster_2_1/xla_runh
?
reduce_window_193*28ž¥@ÀvH€—bcluster_1_1/xla_runh
¦
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28Þô@À{H€Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
Œ
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28œß@ "HÀUbCudnnRNNh*
4
reduce*28àÌ@àvHà›bcluster_0_1/xla_runh
7
	fusion_15*28€Ä@ eH€‘bcluster_1_1/xla_runh
7
	fusion_11*28š¦@ÀYHß—bcluster_1_1/xla_runh
6
reduce_2*28þ˜@€oHà‚bcluster_1_1/xla_runh
6
	fusion_56*28¿ÿ@ÀaHÀzbcluster_0_1/xla_runh
5
fusion_8*28Ê@ÀfHÀubcluster_2_1/xla_runh
5
reduce_2*28Þ»@àbH€gbcluster_0_1/xla_runh
5
fusion_5*28ÿ†@à^H€hbcluster_1_1/xla_runh
3
fusion*28þ¨@ŸZHÀcbcluster_8_1/xla_runh
6
reduce_5*28½‚@ GH€ƒbcluster_1_1/xla_runh
3
fusion*28Ÿ@àYHà`bcluster_3_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ùù@¿IH¿tb%Adam/Adam/update_12/ResourceApplyAdamh
5
reduce_4*28ýõ@€UH fbcluster_1_1/xla_runh
5
fusion_2*28¼‚@ OHÀcbcluster_1_1/xla_runh
>
reduce_window_263*28ÿù@ <H€Ybcluster_1_1/xla_runh
6
	fusion_58*28ßº@àLH€abcluster_0_1/xla_runh
5
fusion_7*28 š@ÀMHÀYbcluster_0_1/xla_runh
6
	fusion_24*28ÿ“@ÿKHàUbcluster_0_1/xla_runh
6
	fusion_43*28Þ@àJHàSbcluster_0_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28¾ñ@ÀHHàVb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_23*28º@€HH€Qbcluster_2_1/xla_runh
4
fusion*28¼Í@€DH€Kbcluster_10_1/xla_runh
5
fusion_9*28Ý¡@à;HŸJbcluster_1_1/xla_runh
6
	fusion_60*28¿”@ @HàGbcluster_0_1/xla_runh
3
fusion*28 è
@ =HàEbcluster_9_1/xla_runh
9
fusion_33__2*28¾Ý
@à2HÀMbcluster_2_1/xla_runh
5
fusion_8*28€§
@à:HÀVbcluster_1_1/xla_runh
6
	fusion_10*28þÿ	@ 1HÀabcluster_1_1/xla_runh
6
	fusion_26*28ßÇ	@€5HàFbcluster_1_1/xla_runh
4
copy_57*28¾“	@€0H€@bcluster_0_1/xla_runh
6
	fusion_61*28¿	@À4H€<bcluster_0_1/xla_runh
6
	fusion_18*28¿î@€2Hà=bcluster_1_1/xla_runh
6
	fusion_16*28þ°@ 0Hÿ6bcluster_2_1/xla_runh
6
	fusion_41*28€¨@à.Hà5bcluster_1_1/xla_runh
6
	fusion_43*28€–@à*HÀ9bcluster_1_1/xla_runh
4
copy_57*28Ÿ“@ *H :bcluster_1_1/xla_runh
4
add_266*28à@à)HÀ<bcluster_2_1/xla_runh
Ž
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28üö@à$H @b%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_30*28¾ð@€-H 2bcluster_2_1/xla_runh
3
fusion*28 â@À%Hà;bcluster_2_1/xla_runh
4
copy_50*28ÞÔ@à*Hà:bcluster_1_1/xla_runh
6
	fusion_42*28€¼@àH Kbcluster_1_1/xla_runh
5
fusion_6*28 ˜@À#HÀ<bcluster_2_1/xla_runh
5
fusion_6*28 ”@à)H€0bcluster_4_1/xla_runh
4
copy_72*28Þ@À(H¿4bcluster_0_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28ÿÜ@€$HÀ6bcluster_1_1/xla_runh
Ã
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ßª@à#H *b
div_no_nanh
6
	fusion_33*28 ¥@à!Hà,bcluster_2_1/xla_runh
6
	fusion_36*28€@€ HÀ+bcluster_2_1/xla_runh
6
	fusion_50*28€…@À HÀ+bcluster_0_1/xla_runh
6
	fusion_21*28ß€@À H .bcluster_2_1/xla_runh
9
fusion_33__1*28Àþ@ H€-bcluster_2_1/xla_runh
6
	fusion_45*28 ù@€H€.bcluster_1_1/xla_runh
5
fusion_1*28Àõ@À H 'bcluster_2_1/xla_runh
6
	fusion_48*28žë@À HÀ)bcluster_2_1/xla_runh
4
add_368*28ÿÅ@à HÀ&bcluster_2_1/xla_runh
6
	fusion_49*28ßµ@€H $bcluster_2_1/xla_runh
6
	fusion_42*28Ÿ›@àH€)bcluster_2_1/xla_runh
4
add_356*28Ÿ‘@€H€,bcluster_2_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À‹@àH€#bAssignAddVariableOp_1h
4
add_343*28ßí@àH "bcluster_2_1/xla_runh
4
add_331*28ÿç@ HÀ"bcluster_2_1/xla_runh
3
add_39*28àÞ@àH  bcluster_4_1/xla_runh
6
	fusion_44*28ÀØ@ H bcluster_1_1/xla_runh
3
fusion*28€Ò@ H  bcluster_5_1/xla_runh
4
slice_1*28 Ï@€H  bcluster_9_1/xla_runh
6
	fusion_27*28¿Ë@€HÀbcluster_2_1/xla_runh
3
fusion*28 ™@€HÀbcluster_7_1/xla_runh
3
add_11*28À•@àHÀbcluster_7_1/xla_runh
3
fusion*28Þ@€Hßbcluster_6_1/xla_runh
´
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ÿù@€HÀbAssignAddVariableOp_7h
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à @à Hà b
LogicalAndh