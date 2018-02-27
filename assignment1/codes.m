x1 = [0,0]; x2 = [-.5, -1]; x3 = [-1, -2];
x4 = [2,0]; x5 = [1.5, -1]; x6 = [1, -2];
x = [x1; x2; x3; x4; x5; x6]
w0 = [1 1]; w1 = [2 -1]; 
w = [w0; w1];

%%
figure(1); clf
plot(x(1:3,1),x(1:3,2),'*')
hold on
plot(x(4:6,1),x(4:6,2),'*')
plot([w(1,1) 0], [w(1,2) 0], 'c-')
plot(w(1,1), w(1,2), 'cd')
text(w(1,1)+.05,w(1,2)-.05,'w_1')
plot([w(2,1) 0], [w(2,2) 0], 'm-')
plot(w(2,1), w(2,2), 'md')
text(w(2,1)+.05,w(2,2)-.05,'w_2')
% plot([w(3,1) 0], [w(3,2) 0], 'k-')
% plot(w(3,1), w(3,2), 'kd')
% text(w(3,1)+.05,w(3,2)-.05,'w_3')

text(x1(1)+.05,x1(2)-.05,'x_1')
text(x2(1)+.05,x2(2)-.05,'x_2')
text(x3(1)+.05,x3(2)-.05,'x_3')
text(x4(1)+.05,x4(2)-.05,'x_4')
text(x5(1)+.05,x5(2)-.05,'x_5')
text(x6(1)+.05,x6(2)-.05,'x_6')
legend('C_1', 'C_2')
xlim([-2 3])
ylim([-3 1])
grid 
rline = refline(-w(1,1)/w(1,2), 0);
rline.LineStyle = '--';
rline.Color = 'c';
rline = refline(-w(2,1)/w(2,2), 0);
rline.LineStyle = '--';
rline.Color = 'm';
% rline = refline(0, 0);
% rline.LineStyle = '--';
% rline.Color = 'k';
% rline = refline(-w(3,2)/w(3,1), 0);
% rline.LineStyle = '--';
% rline.Color = 'k';
legend('C_1', 'C_2')
xlim([-2 3])
ylim([-3 1])
grid minor
s = patch([-1.5 -2 -2 .5], [-3 -3 1 1],'b', 'EdgeColor','magenta');
alpha(s, .1)
s = patch([-1 3 -2 -2 -1], [1 -3 -3 1 1],'b', 'EdgeColor','cyan');
alpha(s, .1)

%% built in solution
net = perceptron;
net = train(net,x',[1 1 1 0 0 0]);
subplot(2,1,2)
plotpv(x', [1 1 1 0 0 0])
plotpc(net.IW{1}, net.b{1})
grid