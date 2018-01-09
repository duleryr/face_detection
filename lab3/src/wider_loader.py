import sys
import os
import numpy as np
import parse_file
import graphical_tools

def get_all_img_info_aux(fd, folder_name, nb_images):
    img_info_vec_aux = []
    for i in range(nb_images):
        img_info_vec_aux.append(parse_file.get_img_info_wider(fd,folder_name))
    return img_info_vec_aux

def get_all_img_info(bbx_file_name):
    fd = open(bbx_file_name)
    img_info_vec = []
    img_info_vec.append(get_all_img_info_aux(fd,"0--Parade",460))
    img_info_vec.append(get_all_img_info_aux(fd,"1--Handshaking",121))
    img_info_vec.append(get_all_img_info_aux(fd,"10--People_Marching",223))
    img_info_vec.append(get_all_img_info_aux(fd,"11--Meeting",130))
    img_info_vec.append(get_all_img_info_aux(fd,"12--Group",582))
    img_info_vec.append(get_all_img_info_aux(fd,"13--Interview",569))
    img_info_vec.append(get_all_img_info_aux(fd,"14--Traffic",76))
    img_info_vec.append(get_all_img_info_aux(fd,"15--Stock_Market",75))
    img_info_vec.append(get_all_img_info_aux(fd,"16--Award_Ceremony",181))
    img_info_vec.append(get_all_img_info_aux(fd,"17--Ceremony",150))
    img_info_vec.append(get_all_img_info_aux(fd,"18--Concerts",213))
    img_info_vec.append(get_all_img_info_aux(fd,"19--Couple",144))
    img_info_vec.append(get_all_img_info_aux(fd,"2--Demonstration",995))
    img_info_vec.append(get_all_img_info_aux(fd,"20--Family_Group",233))
    img_info_vec.append(get_all_img_info_aux(fd,"21--Festival",180))
    img_info_vec.append(get_all_img_info_aux(fd,"22--Picnic",96))
    img_info_vec.append(get_all_img_info_aux(fd,"23--Shoppers",166))
    img_info_vec.append(get_all_img_info_aux(fd,"24--Soldier_Firing",137))
    img_info_vec.append(get_all_img_info_aux(fd,"25--Soldier_Patrol",189))
    img_info_vec.append(get_all_img_info_aux(fd,"26--Soldier_Drilling",156))
    img_info_vec.append(get_all_img_info_aux(fd,"27--Spa",84))
    img_info_vec.append(get_all_img_info_aux(fd,"28--Sports_Fan",184))
    img_info_vec.append(get_all_img_info_aux(fd,"29--Students_Schoolkids",198))
    img_info_vec.append(get_all_img_info_aux(fd,"3--Riot",159))
    img_info_vec.append(get_all_img_info_aux(fd,"30--Surgeons",166))
    img_info_vec.append(get_all_img_info_aux(fd,"31--Waiter_Waitress",202))
    img_info_vec.append(get_all_img_info_aux(fd,"32--Worker_Laborer",178))
    img_info_vec.append(get_all_img_info_aux(fd,"33--Running",105))
    img_info_vec.append(get_all_img_info_aux(fd,"34--Baseball",94))
    img_info_vec.append(get_all_img_info_aux(fd,"35--Basketball",525))
    img_info_vec.append(get_all_img_info_aux(fd,"36--Football",167))
    img_info_vec.append(get_all_img_info_aux(fd,"37--Soccer",219))
    img_info_vec.append(get_all_img_info_aux(fd,"38--Tennis",139))
    img_info_vec.append(get_all_img_info_aux(fd,"39--Ice_Skating",315))
    img_info_vec.append(get_all_img_info_aux(fd,"4--Dancing",172))
    img_info_vec.append(get_all_img_info_aux(fd,"40--Gymnastics",259))
    img_info_vec.append(get_all_img_info_aux(fd,"41--Swimming",324))
    img_info_vec.append(get_all_img_info_aux(fd,"42--Car_Racing",80))
    img_info_vec.append(get_all_img_info_aux(fd,"43--Row_Boat",170))
    img_info_vec.append(get_all_img_info_aux(fd,"44--Aerobics",221))
    img_info_vec.append(get_all_img_info_aux(fd,"45--Balloonist",134))
    img_info_vec.append(get_all_img_info_aux(fd,"46--Jockey",122))
    img_info_vec.append(get_all_img_info_aux(fd,"47--Matador_Bullfighter",244))
    img_info_vec.append(get_all_img_info_aux(fd,"48--Parachutist_Paratrooper",76))
    img_info_vec.append(get_all_img_info_aux(fd,"49--Greeting",164))
    img_info_vec.append(get_all_img_info_aux(fd,"5--Car_Accident",195))
    img_info_vec.append(get_all_img_info_aux(fd,"50--Celebration_Or_Party",196))
    img_info_vec.append(get_all_img_info_aux(fd,"51--Dresses",291))
    img_info_vec.append(get_all_img_info_aux(fd,"52--Photographers",233))
    img_info_vec.append(get_all_img_info_aux(fd,"53--Raid",178))
    img_info_vec.append(get_all_img_info_aux(fd,"54--Rescue",216))
    img_info_vec.append(get_all_img_info_aux(fd,"55--Sports_Coach_Trainer",233))
    img_info_vec.append(get_all_img_info_aux(fd,"56--Voter",198))
    img_info_vec.append(get_all_img_info_aux(fd,"57--Angler",159))
    img_info_vec.append(get_all_img_info_aux(fd,"58--Hockey",178))
    img_info_vec.append(get_all_img_info_aux(fd,"59--people--driving--car",127))
    img_info_vec.append(get_all_img_info_aux(fd,"6--Funeral",170))
    img_info_vec.append(get_all_img_info_aux(fd,"61--Street_Battle",126))
    img_info_vec.append(get_all_img_info_aux(fd,"7--Cheering",163))
    img_info_vec.append(get_all_img_info_aux(fd,"8--Election_Campain",132))
    img_info_vec.append(get_all_img_info_aux(fd,"9--Press_Conference",308))
    img_info_vec = np.concatenate(img_info_vec)
    fd.close()
    return img_info_vec

#img_info_vec = get_all_img_info("../dataset/wider_face_split/wider_face_train_bbx_gt.txt")