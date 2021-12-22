clear all
close all
clc

resp = zeros(20, 44, 8, 9604);

for l = 0:20 %iterate on STYLES
    
    for k = 0:43 %iterate on all CONTENTS
        if exist(['style', num2str(l), '_content', num2str(k), '.jpg'],'file') ~= 0
            %import the image
            disp(['Processing content ', num2str(k), ' in style ', num2str(l)])
            i = double(imresize(imread(['style', num2str(l), '_content', num2str(k), '.jpg']), [512, 512]));
            
            parms.size=5;
            parms.coRelWindowRadius=10;
            parms.numRadiiIntervals=2;
            parms.numThetaIntervals=4;
            parms.varNoise=25*3*36;
            parms.autoVarRadius=1;
            parms.saliencyThresh=0; % I usually disable saliency checking
            parms.nChannels=size(i,3);
            
            radius=(parms.size-1)/2; % the radius of the patch
            marg=radius+parms.coRelWindowRadius;
            
            % Compute descriptor at every 5 pixels seperation in both X and Y directions
            [allXCoords,allYCoords]=meshgrid([marg+1:5:size(i,2)-marg],...
                [marg+1:5:size(i,1)-marg]);
            
            allXCoords=allXCoords(:)';
            allYCoords=allYCoords(:)';
            
            fprintf('Computing self similarity descriptors\n');
            [resp(l+1, k+1, :, :),drawCoords,salientCoords,uniformCoords]=ssimDescriptor(i ,parms ,allXCoords ,allYCoords);
            fprintf('Descriptor computation done\n');
        end
    end
    
end
