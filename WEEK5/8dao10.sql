/*create table team (
id int primary key,
teamname VARCHAR(255)
);

create table score(
id int primary key,
teamid int,
userid int,
score int,
foreign key (teamid) references team(id),
foreign key (userid) references user(id)
);*/

/*insert into team (id,teamname) values (1,'ECNU');
insert into team (id,teamname) values (2,'TOJI');
insert into team (id,teamname) values (3,'SJTU');
INSERT INTO score (id,teamid,userid,score) values (1,1,1,100);
INSERT INTO score (id,teamid,userid,score) values (2,1,4,90);
INSERT INTO score (id,teamid,userid,score) values (3,1,12,66);*/

/*select u.*
from team t
join score s on t.id = s.teamid
join user u on s.userid = u.id
where t.teamname = 'ECNU' and u.age < 20;*/

select coalesce(sum(score),0) as total_score
from team t
join score s on t.id = s.teamid
where t.teamname = 'ECNU';