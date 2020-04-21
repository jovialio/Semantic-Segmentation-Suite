import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import shutil

from utils import utils, helpers
from builders import model_builder

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    alphaValue = 0.5

    if overlay.shape[2] < 4:

        alphaValue = alphaValue * np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype)
        overlay = np.concatenate(
            [
                overlay,
                alphaValue * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def splitInferImage(imagefile, grid_height, grid_width, dataset, model, checkpoint_path):
    print('Processing {}'.format(imagefile))
    class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))

    num_classes = len(label_values)

    print("\n***** Begin prediction *****")
    print("Dataset -->", dataset)
    print("Model -->", model)
    print("Crop Height -->", grid_height)
    print("Crop Width -->", grid_width)
    print("Num Classes -->", num_classes)
    print("Image -->", imagefile)

    # Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

    network, _ = model_builder.build_model(model, net_input=net_input,
                                            num_classes=num_classes,
                                            crop_width=grid_width,
                                            crop_height=grid_height,
                                            is_training=False)

    sess.run(tf.global_variables_initializer())

    print('Loading model checkpoint weights')
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, checkpoint_path)

    print("Testing image " + imagefile)


    outputDirName = os.path.basename(imagefile).split(".")[0] + "_infer"
    parentDir = os.path.dirname(args.inputFile)
    outputDirPath = os.path.join(parentDir,outputDirName)

    if os.path.exists(outputDirPath):
        shutil.rmtree(outputDirPath)
        
    os.mkdir(outputDirPath)

    print('Output dir {}'.format(outputDirPath))
   
    img = cv2.cvtColor(cv2.imread(imagefile,-1), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channels = img.shape

    xScale = img_width // grid_width
    yScale = img_height // grid_height

    outputImage = cv2.cvtColor(img[:yScale*grid_height, :xScale*grid_width,:], cv2.COLOR_RGB2BGR)
    cv2.imwrite("inferenceResult.jpg",outputImage)
    for y in range(0,img_height,grid_height):
        for x in range(0,img_width,grid_width):
           
            if y+grid_height > img_height or x+grid_width > img_width:
                continue
            else:
                resized_image = img[y:y+grid_height,x:x+grid_width]

            input_image = np.expand_dims(np.float32(resized_image),axis=0)/255.0

            st = time.time()
            output_image = sess.run(network,feed_dict={net_input:input_image})

            run_time = time.time()-st

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)

            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
            file_name = utils.filepath_to_name("testing.jpg")
            cv2.imwrite("%s_pred.png"%(file_name),out_vis_image)
            overlay_transparent(outputImage, out_vis_image, x, y)

            print("")
            print("Finished!")
            print("Wrote image " + "%s_pred.png"%(file_name))
            
            # fout_imagefile = outputDirName.split("_")[0] +'_'+ str(x) + '_' + str(y)+".jpg"
            # cv2.imwrite(os.path.join(outputDirPath, fout_imagefile),tiles)

    cv2.imwrite("inferenceResult.jpg",outputImage)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', default="/home/std/data/dronet/SugeiKadut_SCDF/Stitched-AETOS-20191014T025606Z-001/Stitched-AETOS/SKWAY1.jpg", type=str, help = "path to input file")
    parser.add_argument('--gridHeight', default=512, type=int, help = "Grid height")
    parser.add_argument('--gridWidth', default=512, type=int, help = "Grid width")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/0295/model.ckpt", required=False, help='The path to the latest checkpoint weights for your model.')
    parser.add_argument('--model', type=str, default="FC-DenseNet56", required=False, help='The model you are using')
    parser.add_argument('--dataset', type=str, default="dataset/data_dataset_voc", required=False, help='The dataset you are using')
    args = parser.parse_args()

    splitInferImage(args.inputFile, args.gridHeight, args.gridWidth, args.dataset, args.model, args.checkpoint_path)