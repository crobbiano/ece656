function [ z ] = calcZis( etas, x, kernel_mats, Y, samples, ls )
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
    
    % Find class
    z=zeros(1, size(samples,2));
    for i=1:size(samples,2)
        kernel_copy = curr_kernel;
        kernel_copy(:,i) = 0; kernel_copy(i,:) = 0;
        err = [];
        for class=1:num_classes
            err(class) = curr_kernel(i,i) + x(class:class + num_samples_per_class - 1,i)'*...
                kernel_copy(class:class + num_samples_per_class - 1,class:class + num_samples_per_class - 1)*x(class:class + num_samples_per_class - 1,i)...
                - 2*kernel_copy(i,class:class + num_samples_per_class - 1)*x(class:class + num_samples_per_class - 1,i);
        end
        [~, z(i)] = min(err);
    end
    z=z-1;
end

