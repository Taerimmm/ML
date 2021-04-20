
checkpoint_dir = '01_checkpoint'
checkpoint_file = 'VGG19_MaxPool_relu4_4_SpongeBob_ckpt_epoch_80.pth'

# Test image
test_image = "04_test_image/sample_image.png"
output_image = "05_output_image/{}.png".format(checkpoint_file[:-4])

# Test video
source_file = '06_test_video/midnight_in_paris.mp4'
