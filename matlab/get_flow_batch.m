function imo = cnn_ucf101_get_flow_batch(images, inNFrames, imageDir, numThreads)
% get_flow_batch 
% images = 
% inNFrames =
% imageDir =
% numThreads = 
    prefetch = 1;
    imageSize = [224 224 20];
    augmentation = 'none';
    frameSample = 'random';
    nFramesPerVid = 25;
    imReadSz = [224 224 20];
    nStack = imageSize(3);

    sampleFrameLeftRight = floor(nStack/4);
    frameOffsets = [-sampleFrameLeftRight:sampleFrameLeftRight-1]';

    frames = cell(numel(images), nStack, nFramesPerVid);

    sampled_frame_nr = cell(numel(images),1);

    for i=1:numel(images)
        vid_name = images{i};
        nFrames = inNFrames(i);

        % Davide: if the video is too short we have to take all its frames
        if nFrames+1 <= nStack-1
            frameSamples = 1:nFrames-1;

            if length(frameSamples) < nFramesPerVid
                frameSamples = padarray(frameSamples,[0 nFramesPerVid - length(frameSamples)],'symmetric','post');
            elseif length(frameSamples) > nFramesPerVid,
                s = randi(length(frameSamples)-nFramesPerVid);
                frameSamples = frameSamples(s:s+nFramesPerVid-1);
            end

            frameSamples =  repmat(frameSamples, nStack/2,1);                       
        else
            frameSamples = randperm(nFrames-nStack/2)+nStack/4;
            if length(frameSamples) < nFramesPerVid,
                frameSamples = padarray(frameSamples,[0 nFramesPerVid - length(frameSamples)],'symmetric','post');
            elseif length(frameSamples) > nFramesPerVid,
                s = randi(length(frameSamples)-nFramesPerVid);
                frameSamples = frameSamples(s:s+nFramesPerVid-1);
            end

            frameSamples =  repmat(frameSamples,nStack/2,1) +  repmat(frameOffsets,1,size(frameSamples,2));            
        end
        for k = 1:nFramesPerVid
            for j = 1:nStack/2
                frames{i,(j-1)*2+1, k} = ['u' filesep vid_name filesep 'frame' sprintf('%06d.jpg', frameSamples(j,k)) ] ;
                frames{i,(j-1)*2+2, k} = ['v' frames{i,(j-1)*2+1, k}(2:end)];
            end
        end
        sampled_frame_nr{i} = frameSamples;
    end
    frames = strcat([ imageDir filesep], frames);
    if numThreads > 0
        if prefetch
            vl_imreadjpeg(frames, 'numThreads', numThreads, 'prefetch' ) ;
            imo = {frames sampled_frame_nr}  ;
            return ;
        end
    end
end
