function [sharp,pic]=sharppolished(pic)

pic = imsharpen(pic,'Radius',15,'Amount',3.9);

pic2=pic;

pic3=pic;
pic = wiener2(pic,[2 2]);
pic4=pic;
polished1=[-1.2 -0.8 -0.6 ;0 0 0 ;1.2 0.8 0.6 ];
polished11=[1.2 0.8 0.6 ;0 0 0 ;-1.2 -0.8 -0.6 ];
polished111=[-0.1 -0.8 -0.6 ;0 0  0;0.1 0.8 0.6 ];
polished1111=[0.1 0.8 0.6 ;0 0 0 ;-0.1 -0.8 -0.6 ];

polished2=polished1';
polished22=polished11';
polished222=polished111';
polished2222=polished1111';

polished1filter=imfilter(pic,polished1);
polished2filter=imfilter(pic,polished2);
polished11filter=imfilter(pic,polished11);
polished22filter=imfilter(pic,polished22);
polished111filter=imfilter(pic,polished111);
polished222filter=imfilter(pic,polished222);
polished1111filter=imfilter(pic,polished1111);
polished2222filter=imfilter(pic,polished2222);
polished=abs(polished1filter)+abs(polished2filter)+abs(polished11filter)+abs(polished22filter)+abs(polished111filter)+abs(polished222filter)+abs(polished1111filter)+abs(polished2222filter);
sharp=polished;

% [d1,d2] = size(pic);
% for i=1:d1
%     for j=1:d2
%     if polished(i,j)<60
%         sharp(i,j)=0;
%     else
%         sharp(i,j)=polished(i,j);
%     end
%     end
% end

end


