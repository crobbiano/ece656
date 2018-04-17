function [ output_args ] = calcZis( etas, kernel_mats, Y, samples, ls )
    %calcZis Calculates the zi's
    %   Find the current kernel which is a mixture of all ranked kernels
    %   then classifies all samples using the current kernel and
    %   zi == 1 if classification is correct
    
    % get current kernel function
    curr_kernel = zeros(size(kernel_mats{1},1));
    for i=1:length(etas)
        curr_kernel = curr_kernel + etas(i)*kernel_mats{i};
    end
    
    % Find the number of samples in each class
    classes = unique(ls);
    num_classes = numel(classes);
    for i=1:num_classes
        num_samples_per_class(i) = sum(ls == classes(i));
    end
    
    % Find sparse codes for each sample
    x=zeros(size(Y,2), numel(ls));
    for i=1:size(samples,2)
        Yt = Y;
        Yt(:,i) = 0;
        x(:,i) = Yt'*inv(Yt*Yt')*samples(:,i);
    end
    
    % Find class
    z=zeros(1, size(samples,2));
    for i=1:size(samples,2)
        err = [];
        for class=1:num_classes
            err(class) = curr_kernel(i,i) + x(:,i)'*curr_kernel*x(:,i) - 2*curr_kernel(i,:)*x(:,i);
        end
        [~, z(i)] = min(err);
    end
    
end

