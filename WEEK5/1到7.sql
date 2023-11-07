/*3.create table user(
id int not null auto_increment primary key,
name VARCHAR(255),
sex VARCHAR(10),
age INT,
phone VARCHAR(255)
);*/

/*insert into user (name,sex,age,phone) values ('Tom','man',18,'12312321232');
insert into user (name,sex,age,phone) values ('John','man',22,'12353463452');
insert into user (name,sex,age,phone) values ('minnie','woman',18,'124562563452');
insert into user (name,sex,age,phone) values ('lauren','woman',19,'346354634665');
*/

-- select * from user;

-- 4.select * from user where age between 20 and 30;

-- insert into user (name,sex,age,phone) values ('张飞','man',45,'324352345235');
-- insert into user (name,sex,age,phone) values ('张无忌','man',25,'67896785743');
-- select * from user;

/*5.set sql_safe_updates = 0;
delete from user where name like '%张%' and age > 0;
select * from user;
*/

-- 6. select avg(age) as average_age from user;

-- insert into user (name,sex,age,phone) values ('张飞','man',29,'324352345235');
-- insert into user (name,sex,age,phone) values ('张无忌','man',25,'67896785743');

-- 7. select * from user where age between 20 and 30 and name like '%张%' order by age desc;



