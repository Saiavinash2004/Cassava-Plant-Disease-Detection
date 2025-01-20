clc
clear all
close all

[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);

Istrech = imadjust(I,stretchlim(I));
figure(),imshow(Istrech)
title('Contrast stretched image')
%K = medfilt2(Istrech);
%figure(3),imshow(K)

%% Convert RGB image to gray
I1 = rgb2gray(Istrech);
figure(),imshow(I1,[])
title('RGB to gray (contrast stretched) ')
I = imresize(I,[200,200]);

gaussianFilter = fspecial('gaussian',20, 10);
img_filted = imfilter(I1, gaussianFilter,'symmetric');
figure
imshow(img_filted);
title('gaussianFilter Filted Image');
filted_edges = edge(img_filted, 'Canny');
figure();
subplot(121);
imshow(filted_edges);
title('Edges found in filted image');
img_edges = edge(I1, 'Canny');
subplot(122);
imshow(img_edges);
%% Apply median filter to smoothen the image
K = medfilt2(I1);
figure(),imshow(K)
title('median filter')

%MSE and PSNR measurement
[row, col] = size(I);
mse = sum(sum((I(1,1) - K(1,1)).^2)) / (row * col);
psnr = 10 * log10(255 * 255 / mse);

disp('<--------------- Median  filter  ---------------------------->');
disp('Mean Square Error ');
disp(mse);
disp('Peak Signal to Noise Ratio');
disp(psnr);
disp('<--------------------------------------------------------->');
imgID = 2;
HSV=rgb2hsv(I);
figure,imshow(HSV),title('HSV COLOUR TRANSFORM IMAGE');
% %%SEPARATE THREE CHANNELS%%%
H=HSV(:,:,1);
S=HSV(:,:,2);
V=HSV(:,:,3);

figure,imshow(H),title('H-CHANNEL IMAGE');
figure,imshow(S),title('S-CHANNEL IMAGE');
figure,imshow(V),title('V-CHANNEL IMAGE');

figure,
subplot(1,3,1),imshow(H),title('H-CHANNEL');
subplot(1,3,2),imshow(S),title('S-CHANNEL');
subplot(1,3,3),imshow(V),title('V-CHANNEL');

% %% PERFORM RGB TO GRAY CONVERSION ON THE V-CHANNEL IMAGE%%%%
[m n o]=size(V);
if o==3
    gray=rgb2gray(V);
else
    gray=V;
end
figure,imshow(gray);title('V- CHANNEL GRAY IMAGE');

ad=imadjust(gray);
figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
% %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
bw=im2bw(gray,0.5);
figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% 
% % %%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
bw=imcomplement(bw);
figure,imshow(bw);title('COMPLEMENT IMAGE');
% 
% %%%%TO PERFORM MORPHOLOGICAL OPERATIONS IN THE BW IMAGE%%%%
 %%FILL HOLES%%
bw=imfill(bw,'holes');
figure,imshow(bw),title('EDGE BASED SEGMENTATION');
%  %%DILATE OPERATION%%%
SE=strel('square',3);
bw=imdilate(bw,SE);
figure,imshow(bw),title('DILATED IMAGE');
% img=rgb2gray(img); % convert to gray


fontSize = 10;
	redBand = I(:, :, 1);
	greenBand = I(:, :, 2);
	blueBand = I(:, :, 3);
	% Display them.
	figure
	imshow(redBand);
	title('ENCHANCEMENT_1', 'FontSize', fontSize);
	figure
	imshow(greenBand);
	title('ENCHANCEMENT_2', 'FontSize', fontSize);
	figure
	imshow(blueBand);
	title('ENCHANCEMENT_3', 'FontSize', fontSize);
    
tic;


%%
signal1 = feature_ext(I);
    
%% segmentation
GIm = imcomplement(greenBand); 
%Covert RGB to Green Channel Complement
figure,
imshow(GIm);
title('GREEN CHANNEL');

HIm = adapthisteq(GIm);    

%Contrast Limited Adaptive Histogram Equalization
figure,
imshow(HIm);
title('HISTOGRAM ADAPTIVE');

se = strel('ball',8,8);                                                    %Structuring Element
gopen = imopen(HIm,se);                                                    %Morphological Open
godisk = HIm - gopen;                                                      %Remove Optic Disk

medfilt = medfilt2(godisk);                                                %2D Median Filter
background = imopen(medfilt,strel('disk',100));                            %imopen function
I2 = medfilt - background;                                                 %Remove Background
GC = imadjust(I2);                                                         %Image Adjustment
figure,
imshow(GC);
title('ADJUST IMAGES');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------- Segmentation Using Fuzzy C-means(fcm) ---------------- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

IM=GC;
IM=double(IM);
[maxX,maxY]=size(IM);
IMM=cat(3,IM,IM);

cc1=8;
cc2=250;

ttFcm=0;

while(ttFcm<10)
    ttFcm=ttFcm+1;
    
    sttFcm=(['ttFcm = ' num2str(ttFcm)]);
   
    
    c1=repmat(cc1,maxX,maxY);
    c2=repmat(cc2,maxX,maxY);
    
    if ttFcm==1 
        test1=c1; test2=c2;
    end
    
    c=cat(3,c1,c2);
    ree=repmat(0.000001,maxX,maxY);
    ree1=cat(3,ree,ree);
    
    distance=IMM-c;
    distance=distance.*distance+ree1;
    
    daoShu=1./distance;
    
    daoShu2=daoShu(:,:,1)+daoShu(:,:,2);
    distance1=distance(:,:,1).*daoShu2;
    u1=1./distance1;
    distance2=distance(:,:,2).*daoShu2;
    u2=1./distance2;
      
    ccc1=sum(sum(u1.*u1.*IM))/sum(sum(u1.*u1));
    ccc2=sum(sum(u2.*u2.*IM))/sum(sum(u2.*u2));
   
    tmpMatrix=[abs(cc1-ccc1)/cc1,abs(cc2-ccc2)/cc2];
    pp=cat(3,u1,u2);
    
    for i=1:maxX
        for j=1:maxY
            if max(pp(i,j,:))==u1(i,j)
                IX2(i,j)=1;
            else
                IX2(i,j)=2;
            end
        end
    end
    
    if max(tmpMatrix)<0.0001
        break;
    else
        cc1=ccc1;
        cc2=ccc2;
    end

    for i=1:maxX
        for j=1:maxY
            if IX2(i,j)==2
                IMMM(i,j)=254;
            else
                IMMM(i,j)=8;
            end
        end
    end
    
    background=imopen(IMMM,strel('disk',45));
    I4=IMMM-background;
    I4=bwareaopen(I4,30);
    figure,
imshow(I4);
title('FCM');
 



for i=1:maxX
    for j=1:maxY
        if IX2(i,j)==2
            IMMM(i,j)=200;
        else
            IMMM(i,j)=1;
        end
    end
end 

ffcm1=(['The 1st Cluster = ' num2str(ccc1)]);
ffcm2=(['The 2nd Cluster = ' num2str(ccc2)]);
[m,n]=size(I4);
Tn=0;
Tp=0;
Fp=0;
Fn=0;

for i=1:m
    for j=1:n
        if I4(i,j)==0 && I(i,j)==0 
            Tn=Tn+1;
        elseif I4(i,j)==1 && I(i,j)==1
            Tp=Tp+1;
        elseif  I4(i,j)==1 && I(i,j)==0
            Fp=Fp+1;
        elseif  I4(i,j)==0 && I(i,j)==1
            Fn=Fn+1;
        end
    end
end
aucc=(Tp+Tn)/(Tp+Tn+Fp+Fn);                                                %Accuracy                                                 
sensitivity=Tp/(Tp+Fn);                                                    %True Positive Rate
specificity=Tn/(Tn+Fp);                                                    %True Negative Rate
fpr=1-specificity;                                                         %False Positive Rate
ppv=Tp/(Tp+Fp);                                                            %Positive Predictive Value
disp('True Positive = ');
disp(Tp);
disp('True Negative = ');
disp(Tn);
disp('False Positive = ');
disp(Fp);
disp('False Negative = ');
disp(Fn);
disp('False Positive Rate = ');
disp(fpr);
disp('Sensitivity = ');
disp(sensitivity);
disp('Specificity = ');
disp(specificity);
disp('Accuracy = ');
disp(aucc);
disp('Positive Predictive Value = ');
disp(ppv);

%%
end
cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end


figure, subplot(3,1,1);imshow(segmented_images{1});title('ROI based segment'); subplot(3,1,2);imshow(segmented_images{2});title('segmentation');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));


Img = double(I(:,:,1));
epsilon = 1;
switch imgID

    case 1
        num_it =1000;
        rad = 8;
        alpha = 0.3;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(15:78,32:95) = 1;
        seg = local_AC_MS(Img,mask_init,rad,alpha,num_it,epsilon);
    case 2
        num_it =800;
        rad = 9;
        alpha = 0.003;% coefficient of the length term
        mask_init = zeros(size(Img(:,:,1)));
        mask_init(53:77,56:70) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
    case 3
        num_it = 1500;
        rad = 5;
        alpha = 0.001;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(47:80,86:99) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
end

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);

g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
% m = size(G,1);
% n = size(G,2);
% in_diff = 0;
% for i = 1:m
%     for j = 1:n
%         temp = G(i,j)./(1+(i-j).^2);
%         in_diff = in_diff+temp;
%     end
% end
% IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness];
disp('-----------------------------------------------------------------');
disp('Contrast = ');
disp(Contrast);
disp('Correlation = ');
disp(Correlation);
disp('Energy = ');
disp(Energy);
disp('Mean = ');
disp(Mean);
disp('Standard_Deviation = ');
disp(Standard_Deviation);
disp('Entropy = ');
disp('RMS = ');
disp(Entropy);
disp(RMS);
disp('Variance = ');
disp(Variance);
disp('Kurtosis = ');
disp(Kurtosis);
disp('Skewness = ');
disp(Skewness);
load Trainset.mat
 xdata = meas;
 group = label;
acc=accuracy_image(feat); 
disp(acc);
addpath('../../Training' , '../../mdANN' , '../../utilCode' );
testData = load('data/ann_dataset_trained.mat'); 
addpath('lib')
nnOptions = {};

% % Alternative options
% nnOptions = {'lambda', 0.1,...
%             'maxIter', 50,...
%             'hiddenLayers', [40 20],...
%             'activationFn', 'tanh',...
%             'validPercent', 30,...
%             'doNormalize', 1};

%% Learning
modelNN = learnNN(testData.X, testData.y, nnOptions);
% plotting the confusion matrix for the validation set
figure(); cla(gca);
plotConfMat(modelNN.confusion_valid);

%% Predicting on a random image
rI = randi(size(testData.X, 1)); % a random index
p = predictNN(testData.X(rI,:), modelNN); % the prediction

figure(); cla(gca);
imagesc(reshape(testData.X(rI,:), 20, 20)); % plotting
% colormap(flipud(gray));
% title(sprintf('Actual: %d, Predicted: %d', ...
%     mod(testData.y(rI), 10), mod(p, 10))); % index for number 0 is 10

numTest=300;numTrain=100;
x=randi(16,1,numTrain+numTest)-1; 
xBin=[mod(x,2) ;mod(floor(x/2),2) ;mod(floor(x/4),2) ;mod(floor(x/8),2)]; 

% 'hide' the 4 bits inside a larger vector padded with random bits and fixed bits
samples = [repmat((1:10)',1,size(xBin,2)) ; xBin ; rand(10,size(xBin,2))/2*mean(xBin(:))]; 

dataset=[];
for idx=1:size(samples,2)
    if (idx>numTrain)
        dataset.I_test{idx-numTrain} = samples(:,idx-numTrain);
        dataset.labels_test(idx-numTrain)=x(idx-numTrain);
    else
    dataset.I{idx} = samples(:,idx);
    dataset.labels(idx)=x(idx);
    end
end

net = CreateNet('../../Configs/1d.conf');  % small 1d fully connected net,will converge faster

net   =  Train(dataset,net, 100);

checkNetwork(net,Inf,dataset,1);

result = ann_classifier(feat,meas,label);
helpdlg(result);
load('Accuracy_Data.mat')
Accuracy_Percent= zeros(200,1);
for i = 1:800
data = Train_Feat;
groups = ismember(Train_Label,1);
% groups = ismember(Train_Label,0);
[train,feat] = crossvalind('HoldOut',groups);
cp = classperf(groups);
 classperf(cp,feat);
Accuracy = cp.CorrectRate*2;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of cnn with 800 iterations is: %g%%',Max_Accuracy)



% warning('on','all');
	fontSize = 10;
	
	% Compute and plot the red histogram.
	hR = figure
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
	bar(countsR, 'r');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('GA GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	hG = figure
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
	bar(countsG, 'g', 'BarWidth', 0.95);
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
    title('GA GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	hB = figure
	[countsB, grayLevelsB] = imhist(blueBand);
	maxGLValueB = find(countsB > 0, 1, 'last');
	maxCountB = max(countsB);
	bar(countsB, 'b');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('GA GRAPH', 'FontSize', fontSize);
	
	% Set all axes to be the same width and height.
	% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]);
% 	if eightBit
% 		maxGL = 255;
% 	end
	maxCount = max([maxCountR,  maxCountG, maxCountB]);
% 	axis([hR hG hB], [0 maxGL 0 maxCount]);
	
	% Plot all 3 histograms in one plot.
	figure
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('ALL OVER GA REGION', 'FontSize', fontSize);



if(strcmp(result,'Pepper__bell___Bacterial_spot'))
     
xi = linspace(-6,2,100);
yi = linspace(-4,4,50);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'PEPPER BELL AFFECTED PLEASE BE CURE THIS DISEASE';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);

      

end
   
if(strcmp(result,'Pepper__bell___healthy'))
     
xi = linspace(-6,2,200);
yi = linspace(-4,4,510);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'HEALTHY LEAF';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      

end
if(strcmp(result,'Potato___Early_blight'))
    
xi = linspace(-6,2,20);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'POTATO BLIGHT DISEASE AFFECTED';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
       

end
if(strcmp(result,'Potato___healthy'))
     
xi = linspace(-6,2,20);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'HEALTHY LEAF';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
        

end
if(strcmp(result,'Tomato__Tomato_Yellow Leaf__Curl_Virus'))
     
xi = linspace(-6,2,20);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'TOMATO LEAF CURL DISEASE AFFECTED';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      

end
if(strcmp(result,'Tomato_Late_blight'))
       
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'TOMATO LATE BLIGHT';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
     
end
if(strcmp(result,'Cercospora Leaf Spot'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm') 
defaultString = 'AFFECTED LEAF DISEASE';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
     
end
if(strcmp(result,'Grape Black_rot'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'BLACK ROT DISEASE';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      
end
if(strcmp(result,'Grape Esca_(Black_Measles)'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'AFFECTED DISEASES';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
     
end

if(strcmp(result,'Cassava bacterial blight(cbb)'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'Cassava bacterial blight(cbb)';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
    
end

if(strcmp(result,'Brown_Spot'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'Brown_Spot';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      
end

if(strcmp(result,'Mosaic Disease'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'Mosiac Disease';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      
end

if(strcmp(result,'Green_Mottel'))     
xi = linspace(-6,2,100);
yi = linspace(-4,4,50);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')
defaultString = 'Green Motel AFFECTED PLEASE BE CURE THIS DISEASE';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);

end

if(strcmp(result,'Cassava Healthy'))
xi = linspace(-6,2,80);
yi = linspace(-4,4,80);
[X,Y] = meshgrid(xi,yi);
Z = ps_example([X(:),Y(:)]);
Z = reshape(Z,size(X));
surf(X,Y,Z,'MeshStyle','none')
colormap 'jet'
view(-26,43)
xlabel('Fitness level')
ylabel('Healthy level')
title('Genetic Algorithm')   
defaultString = 'Cassava Healthy';
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, defaultString);
      
end
