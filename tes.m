%% Training Neural Gas Network
for i = 1 : filesnumber(1,1)
[L{i},Centers{i}] = imsegkmeans(X{i},5);
KM{i} = labeloverlay(X{i},L{i});
disp(['Loading image No :   ' num2str(i) ]);
end;

