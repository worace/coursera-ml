function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%PLOTDATA(x,y) plots the data points with + for the positive examples
%and o for the negative examples. X is assumed to be a Mx2 matrix.

figure; hold on;

admit = find(y==1);
reject = find(y==0);

plot(X(admit,1), X(admit,2),"k+",'LineWidth', 2,'MarkerSize', 7);
plot(X(reject,1),X(reject,2),"ko",'MarkerFaceColor', 'y','MarkerSize', 7);
xlabel('Exam 1 Score');
ylabel("Exam 2 Score");
legend('Admitted', 'Not admitted')

hold off;

end
