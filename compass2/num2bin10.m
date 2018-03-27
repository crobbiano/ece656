function [ binnum ] = num2bin10( num )
%num2bin10 Converts 0-9 to 10 digit fake binary
% num array of numbers

binnum = zeros(10, numel(num));

for i = 1:numel(num)
    binnum(num(i)+1, i) = 1; 
end

end

