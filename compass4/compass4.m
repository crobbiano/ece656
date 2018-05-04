%% Compass 4 ece656 SOFM
clear all
clc
close all
%% Read in the required images
load('BABOON.mat')
load('Boat.mat')
load('Lena.mat')
load('Peppers.mat')
%% Use one of the images and deconstruct into 4x4 cells
stride = 4;
num_cells_w = 512/stride;
num_cells_h = 512/stride;
images = {baboon, boat, lena, peppers};


for im_num=1:4
    im = images{im_num};
    cell_cnt = 1;
    for i=1:num_cells_w
        for j=1:num_cells_h
            image_cells{cell_cnt} = im((i-1)*stride + 1: (i-1)*stride + stride,(j-1)*stride + 1: (j-1)*stride + stride);
            image_vecs{im_num}(:,cell_cnt) = reshape(image_cells{cell_cnt}, [], 1);
            cell_cnt = cell_cnt + 1;
        end
    end
end

%% Make SOFMs and train
P1 = image_vecs{1};
net_num = 1;
maps = {[4 4], [4 8], [8 8], [8 16], [16 16], [16 32], [32 32]};
% for n=1:length(maps)
for n=1:1
    nets{net_num} = selforgmap(maps{n});
    nets{net_num} = configure(nets{net_num},P1);
    nets{net_num}.trainParam.epochs = 500;
    nets{net_num} = train(nets{net_num},P1);
    features{net_num} = nets{net_num}.IW{1}';
    net_num = net_num + 1;
end
%% Reconstruct
for net_num=1:numel(nets)
    for im_num=1:4
        y{net_num}{im_num}=nets{net_num}(image_vecs{im_num});
        classes{net_num}{im_num}=vec2ind(y{net_num}{im_num});
        
        cell_cnt = 1;
        for i=1:num_cells_w
            for j=1:num_cells_h
                im_recon{net_num}{im_num}((i-1)*stride + 1: (i-1)*stride + stride,(j-1)*stride + 1: (j-1)*stride + stride) = reshape(features{net_num}(:,classes{net_num}{im_num}(cell_cnt)),stride,stride);
                cell_cnt = cell_cnt + 1;
            end
        end
        [~,snrs{net_num}{im_num}] = psnr(im_recon{net_num}{im_num}, images{im_num});
        ssims{net_num}{im_num} = ssim(im_recon{net_num}{im_num}, images{im_num});
    end
end

%% Caculate some things and Plot some things
snrplot = [];
bppplot = [];
for i=1:length(maps)
    bpp(i) = log2(maps{i}(1)*maps{i}(2)) / (stride*stride);
    
    feature_num = i;
    figure(i); clf;
    subplot(2, 4, 1)
    imshow(im_recon{feature_num}{1},[])
    xlabel(ssims{i}{1})
    subplot(2, 4, 5)
    imshow(images{1},[])
    subplot(2, 4, 2)
    imshow(im_recon{feature_num}{2},[])
    xlabel(ssims{i}{2})
    subplot(2, 4, 6)
    imshow(images{2},[])
    subplot(2, 4, 3)
    imshow(im_recon{feature_num}{3},[])
    xlabel(ssims{i}{3})
    subplot(2, 4, 7)
    imshow(images{3},[])
    subplot(2, 4, 4)
    imshow(im_recon{feature_num}{4},[])
    xlabel(ssims{i}{4})
    subplot(2, 4, 8)
    imshow(images{4},[])
    str = sprintf("%u neurons", (maps{i}(1)*maps{i}(2)));
    suptitle(str)
    
    thing1 = snrs{i}(:);
    thing2 = repmat(bpp(i), [1 length(thing1)])';
    snrplot = [snrplot; thing1];
    bppplot = [bppplot; thing2];
end
snrplot = cell2mat(snrplot);
figure(17); clf;
hold on
plot(bppplot, snrplot, '.')
hold off

% for storage -> comrpession rate = image size / block size; in naive sense
% for communication -> 8bpp * num_pixels turns into num_blocks*log2(num_features)
compression_rate = 512 / 16
compression_rate_real = (8*512*512) / (((512*512)/(stride*stride)) * log2(128))

% Reconstructed images look good around 2^10 features
