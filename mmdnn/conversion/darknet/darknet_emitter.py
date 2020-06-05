
import os
import json
import re
import argparse
import shutil


conv_layers = []

no_yolo_layers = 0

def find_index(layer_name):
    layer_name = layer_name.split('-')[0]
    for idx, layer in enumerate(conv_layers):
        if layer.split('-')[0] == layer_name:
            return idx
    return -1

def get_convolution_layer(attrs, name, inputs):
    global no_yolo_layers
    out_temp = ''
    if len(conv_layers) > 0:
        if len(inputs) == 1:
            last_layer = re.findall(r"[0-9]+", conv_layers[-1])
            input_layer = re.findall(r"[0-9]+", inputs[0])

            # print('layer {} has input {}, last layer is {}'.format(
            #     name, inputs[0], input_layer))

            if 'maxpool' in last_layer and int(last_layer[0]) != int(input_layer[0]) -1:
                out_temp += "\n[route] \nlayers = {}".format(find_index(inputs[0]) - len(conv_layers))
            elif int(last_layer[0]) != int(input_layer[0]):
                out_temp += "[route] \nlayers = {}".format(find_index(inputs[0]) - len(conv_layers) - 1)
            out_temp += '\n\n'


    dims = attrs["_output_shape"]["list"]["shape"][0]["dim"]
    size=attrs["kernel_shape"]["list"]["i"]
    stride=attrs["strides"]["list"]["i"]
    padding=attrs["pads"]["list"]["i"]
    out = out_temp + "[convolutional] \nstride={} \nsize={} \nfilter={} \npad={}\n".format(stride[1],size[1],dims[3]["size"],padding[2])


    conv_layers.append(name)
    return out

def get_input_layer(attrs):
    dims = attrs["_output_shape"]["list"]["shape"][0]["dim"]
    out ="[net]\nbatch = 64 \nsubdivisions=2\n"
    out += "width={} \nheight={} \nchannels={}\n".format(dims[1]["size"], dims[2]["size"], dims[3]["size"],)
    out += "momentum={} \ndecay={}\nangle={} \nsaturation = {} \nexposure = {} \nhue = .{}\n".format(
        0.9, 0.0005,  0, 1.5, 1.5, 1
    )
    out += 'learning_rate=0.001 \nburn_in = 1000 \nmax_batches = 500200 ' \
           '\npolicy = steps \nsteps=400000,450000 \nscales=.1,.1\n'

    out += '\n'
    return out

def get_batch_norm_layer():
    return "batch_normalize=1 \n"


def get_leaky_relu():
    return "activation=leaky \n\n"

def parseJson(list):
    elems=''
    for idx, m in enumerate(list):
        if idx != len(list) - 1:
            elems += "{},".format(m)
        else:
            elems += "{}".format(m)
    return elems


def get_yolo(attrs, name):
    global no_yolo_layers
    mask_o=parseJson(attrs["mask"]["list"]["i"])
    anchors_nr=parseJson(attrs["anchors"]["list"]["i"])
    jitter_nr=str(attrs["jitter"]["f"]).split('.')[1]
    ignore_threshold_nr = str(attrs["ignore_thresh"]["f"]).split('.')[1]
    truth_threshold_nr = str(attrs["truth_thresh"]["f"]).split('.')[0]
    random_nr = str(attrs["random"]["f"]).split('.')[0]
    out = '\n[yolo] \nmask = {} \nanchors = {} \nclasses = {} \nnum = {} \njitter=.{}' \
          '\nignore_thresh = .{} \ntruth_thresh = {} \nrandom = {}\n\n'.format(
        mask_o,anchors_nr,attrs["classes"]["i"],attrs["num"]["i"],jitter_nr,ignore_threshold_nr,truth_threshold_nr,random_nr
    )
    no_yolo_layers += 1
    return out

def get_upsample(attrs, name):
    conv_layers.append(name)
    stride = attrs["scales"]["list"]["i"]
    out = "\n[upsample]\nstride={}\n".format(stride[0])

    # out += '\n'
    return out

def get_max_pool(attrs, name):
    conv_layers.append(name)
    stride = attrs["strides"]["list"]["i"]
    size = attrs["kernel_shape"]["list"]["i"]
    out = "[pool]\nstride={} \nsize={} \n".format(stride[1], size[1])

    # out += '\n'
    return out

def get_concat(attrs, name, inputs):
    out = "\n[route] \nlayers = "
    conv_layers.append(name)
    for idx,input in enumerate(inputs):
        if idx != len(inputs) -1:
            out += "{},".format(find_index(input) - len(conv_layers) + 1)
        else:
            out += "{}".format(find_index(input) - len(conv_layers) + 1)
    out += '\n'
    return out


def main(model_path, weights_path, outdir):
    print("path to json:{}".format(model_path))
    out=""
    with open(model_path) as fh:
        content = json.load(fh)
    for idx, layer in enumerate(content["node"]):
        op = layer["op"]
        try:
            if op == "DataInput":
                out+=get_input_layer(layer["attr"])
            elif op=="Conv":
                out+=get_convolution_layer(layer["attr"], layer["name"], layer["input"] )
            elif op=="BatchNorm":
                out+=get_batch_norm_layer()
            elif op=="LeakyRelu":
                out+=get_leaky_relu()
            elif op=="Pool":
                out+=get_max_pool(layer["attr"], layer["name"])
            elif op=="UpSampling2D":
                out += get_upsample(layer["attr"], layer["name"])
            elif op=='yolo':
                out+=get_yolo(layer["attr"], layer["name"])
            elif op=='Concat':
                out+=get_concat(layer["attr"], layer["name"], layer["input"] )
        except Exception as e:
            print('[ERROR] Could not transform layer {}'.format(op))
    with open(os.path.join(outdir, "darknet.cfg"),"w") as fh:
        fh.write(out)
    print('[INFO] created darknet.cfg in:{}'.format(os.path.join(outdir, "darknet.cfg")))

    shutil.copy(weights_path, os.path.join(outdir,'darknet.weights'))
    print('[INFO] created darknet.weights in:{}'.format(os.path.join(outdir,'darknet.weights')))

    print('[INFO] Done conversion')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform IR model to darknet')
    parser.add_argument('--model_path', '-m', type=str, help='Input model path')
    parser.add_argument('--weights_path', '-w', type=str, help='Input weights path')
    parser.add_argument('--outdir', '-o', type=str, help='Output dir for darknet model')
    args = parser.parse_args()

    # path_to_json=r"D:\Programming\Output\.json"
    main(args.model_path, args.weights_path, args.outdir)
