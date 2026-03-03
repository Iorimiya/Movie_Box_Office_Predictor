/*
 Navicat Premium Dump SQL

 Source Server         : Local Server
 Source Server Type    : MariaDB
 Source Server Version : 110802 (11.8.2-MariaDB-ubu2404)
 Source Host           : localhost:27040
 Source Schema         : movie_data

 Target Server Type    : MariaDB
 Target Server Version : 110802 (11.8.2-MariaDB-ubu2404)
 File Encoding         : 65001

 Date: 05/02/2026 00:38:20
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for box_office
-- ----------------------------
DROP TABLE IF EXISTS `box_office`;
CREATE TABLE `box_office`  (
  `on_air_week_id` int(11) NOT NULL COMMENT '上映週ID',
  `amount` bigint(20) NULL DEFAULT NULL COMMENT '票房金額',
  PRIMARY KEY (`on_air_week_id`) USING BTREE,
  INDEX `idx_week_amount`(`on_air_week_id` ASC, `amount` ASC) USING BTREE,
  CONSTRAINT `fk_box_office_on_air_week` FOREIGN KEY (`on_air_week_id`) REFERENCES `on_air_weeks` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_uca1400_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for movies
-- ----------------------------
DROP TABLE IF EXISTS `movies`;
CREATE TABLE `movies`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '電影ID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NULL DEFAULT NULL COMMENT '電影名稱',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_uca1400_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for on_air_weeks
-- ----------------------------
DROP TABLE IF EXISTS `on_air_weeks`;
CREATE TABLE `on_air_weeks`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '上映週ID',
  `movie_id` int(11) NOT NULL COMMENT '電影ID',
  `start_date` date NOT NULL COMMENT '該週週日的日期',
  `year_week` int(6) GENERATED ALWAYS AS (yearweek(`start_date`,0)) VIRTUAL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_movie_week`(`movie_id` ASC, `start_date` ASC) USING BTREE,
  CONSTRAINT `fk_on_air_movie` FOREIGN KEY (`movie_id`) REFERENCES `movies` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_uca1400_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for replies
-- ----------------------------
DROP TABLE IF EXISTS `replies`;
CREATE TABLE `replies`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '回覆ID',
  `review_id` int(11) NOT NULL COMMENT '評論文章ID',
  `type` enum('push','boo','arrow') CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NOT NULL DEFAULT 'arrow' COMMENT '推、噓或箭頭',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NULL DEFAULT NULL COMMENT '回覆內容',
  `created_at` datetime NOT NULL DEFAULT current_timestamp() COMMENT '建立時間(日期+時間)',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_reply`(`review_id` ASC, `created_at` ASC, `content`(100) ASC) USING BTREE,
  INDEX `idx_rev_id`(`review_id` ASC) USING BTREE,
  INDEX `idx_created_at`(`created_at` ASC) USING BTREE,
  CONSTRAINT `fk_reply_rev` FOREIGN KEY (`review_id`) REFERENCES `reviews` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_uca1400_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for reviews
-- ----------------------------
DROP TABLE IF EXISTS `reviews`;
CREATE TABLE `reviews`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '評論文章ID',
  `movie_id` int(11) NOT NULL COMMENT '電影ID',
  `url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NOT NULL COMMENT '評論網址',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NULL DEFAULT NULL COMMENT '評論標題',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NOT NULL COMMENT '評論內容',
  `created_at` date NULL DEFAULT NULL COMMENT '建立時間(日期)',
  `type` enum('public','expert') CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_ai_ci NOT NULL DEFAULT 'public' COMMENT '種類(大眾評論或專家評論)',
  `sentiment_score` double NULL DEFAULT NULL COMMENT '情緒分數',
  `expert_score` double NULL DEFAULT NULL COMMENT '專家評分',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_url`(`url` ASC) USING BTREE,
  INDEX `fk_rev_movie`(`movie_id` ASC, `type` ASC) USING BTREE,
  INDEX `idx_review_date`(`created_at` ASC) USING BTREE,
  INDEX `idx_review_type`(`type` ASC) USING BTREE,
  INDEX `idx_movie_created`(`movie_id` ASC, `created_at` ASC) USING BTREE,
  CONSTRAINT `fk_rev_movie` FOREIGN KEY (`movie_id`) REFERENCES `movies` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `chk_sentiment` CHECK (`sentiment_score` <= 1 and `sentiment_score` >= 0)
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_uca1400_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- View structure for counts
-- ----------------------------
DROP VIEW IF EXISTS `counts`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `counts` AS with review_count_2023_2025 as (select count(0) AS `review_count_2023_2025` from (`movie_id_2023_2025` `mi` join `movie_reviews` `mrv` on(`mrv`.`movie_id` = `mi`.`movie_id`))), review_count as (select count(0) AS `review_count` from `movie_reviews`)select `review_count_2023_2025`.`review_count_2023_2025` AS `review_count_2023_2025`,`review_count`.`review_count` AS `review_count` from (`review_count_2023_2025` join `review_count`);

-- ----------------------------
-- View structure for movie_box_office
-- ----------------------------
DROP VIEW IF EXISTS `movie_box_office`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `movie_box_office` AS select `m`.`id` AS `movie_id`,`m`.`name` AS `movie_name`,`oaw`.`start_date` AS `start_date`,`oaw`.`start_date` + interval 6 day AS `end_date`,`bo`.`amount` AS `amount` from ((`movies` `m` join `on_air_weeks` `oaw` on(`m`.`id` = `oaw`.`movie_id`)) join `box_office` `bo` on(`oaw`.`id` = `bo`.`on_air_week_id`)) order by `m`.`id`;

-- ----------------------------
-- View structure for movie_id_2023_2025
-- ----------------------------
DROP VIEW IF EXISTS `movie_id_2023_2025`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `movie_id_2023_2025` AS select `oaw`.`movie_id` AS `movie_id` from `on_air_weeks` `oaw` group by `oaw`.`movie_id` having min(`oaw`.`start_date`) between '2023-01-01' and '2025-12-31';

-- ----------------------------
-- View structure for movie_list_2023_2025
-- ----------------------------
DROP VIEW IF EXISTS `movie_list_2023_2025`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `movie_list_2023_2025` AS select `mi`.`movie_id` AS `id`,`m`.`name` AS `name` from (`movies` `m` join `movie_id_2023_2025` `mi` on(`m`.`id` = `mi`.`movie_id`));

-- ----------------------------
-- View structure for movie_replies
-- ----------------------------
DROP VIEW IF EXISTS `movie_replies`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `movie_replies` AS select `mrv`.`movie_id` AS `movie_id`,`mrv`.`review_id` AS `review_id`,`rep`.`id` AS `reply_id`,`rep`.`type` AS `type`,`rep`.`content` AS `content`,`rep`.`created_at` AS `created_at` from (`movie_reviews` `mrv` join `replies` `rep` on(`rep`.`review_id` = `mrv`.`review_id`));

-- ----------------------------
-- View structure for movie_reviews
-- ----------------------------
DROP VIEW IF EXISTS `movie_reviews`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `movie_reviews` AS select `m`.`id` AS `movie_id`,`rev`.`id` AS `review_id`,`rev`.`url` AS `url`,`rev`.`title` AS `title`,`rev`.`content` AS `content`,`rev`.`created_at` AS `created_at`,`rev`.`type` AS `type`,`rev`.`sentiment_score` AS `sentiment_score`,`rev`.`expert_score` AS `expert_score` from (`movies` `m` join `reviews` `rev` on(`m`.`id` = `rev`.`movie_id`));

-- ----------------------------
-- View structure for on_air_weeks_detail_view
-- ----------------------------
DROP VIEW IF EXISTS `on_air_weeks_detail_view`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `on_air_weeks_detail_view` AS select `on_air_weeks`.`id` AS `on_air_weeks_id`,`on_air_weeks`.`movie_id` AS `movie_id`,`on_air_weeks`.`start_date` AS `start_date`,`on_air_weeks`.`start_date` + interval 6 day AS `end_date` from `on_air_weeks` order by `on_air_weeks`.`start_date` desc;

-- ----------------------------
-- View structure for weekly_feature_data
-- ----------------------------
DROP VIEW IF EXISTS `weekly_feature_data`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `weekly_feature_data` AS with review_stats as (select `weekly_reviews`.`on_air_week_id` AS `week_id`,sum(case when `weekly_reviews`.`review_type` = 'public' then 1 else 0 end) AS `public_review_count`,sum(case when `weekly_reviews`.`review_type` = 'public' then `weekly_reviews`.`content_length` else 0 end) AS `total_content_length`,sum(case when `weekly_reviews`.`review_type` = 'public' then `weekly_reviews`.`title_length` else 0 end) AS `total_title_length`,sum(case when `weekly_reviews`.`review_type` = 'expert' then 1 else 0 end) AS `expert_review_count`,avg(case when `weekly_reviews`.`review_type` = 'expert' then `weekly_reviews`.`expert_score` end) AS `average_expert_score`,sum(`weekly_reviews`.`sentiment_score`) AS `total_sentiment_score`,count(`weekly_reviews`.`sentiment_score`) AS `count_sentiment_score` from `weekly_reviews` group by `weekly_reviews`.`on_air_week_id`), reply_stats as (select `wr`.`on_air_week_id` AS `week_id`,count(`rp`.`id`) AS `total_reply_count`,sum(case when `rp`.`type` = 'push' then 1 else 0 end) AS `total_positive_reactions`,sum(case when `rp`.`type` = 'boo' then 1 else 0 end) AS `total_negative_reactions`,sum(case when `rp`.`created_at` >= `wr`.`week_start_date` and `rp`.`created_at` < `wr`.`week_start_date` + interval 7 day then 1 else 0 end) AS `weekly_total_reply_count`,sum(case when `rp`.`type` = 'push' and `rp`.`created_at` >= `wr`.`week_start_date` and `rp`.`created_at` < `wr`.`week_start_date` + interval 7 day then 1 else 0 end) AS `weekly_positive_reply_count`,sum(case when `rp`.`type` = 'boo' and `rp`.`created_at` >= `wr`.`week_start_date` and `rp`.`created_at` < `wr`.`week_start_date` + interval 7 day then 1 else 0 end) AS `weekly_negative_reply_count` from (`weekly_reviews` `wr` join `replies` `rp` on(`rp`.`review_id` = `wr`.`review_id`)) where `wr`.`review_type` = 'public' group by `wr`.`on_air_week_id`)select `oaw`.`id` AS `on_air_weeks_id`,`oaw`.`movie_id` AS `movie_id`,`m`.`name` AS `movie_name`,`oaw`.`start_date` AS `start_date`,`oaw`.`start_date` + interval 6 day AS `end_date`,coalesce(`rs`.`total_content_length`,0) AS `total_content_length`,coalesce(`rs`.`total_title_length`,0) AS `total_title_length`,`bo`.`amount` AS `box_office`,`rs`.`total_sentiment_score` / nullif(`rs`.`count_sentiment_score`,0) AS `average_sentiment_score`,`rs`.`average_expert_score` AS `average_expert_score`,coalesce(`rs`.`public_review_count`,0) + coalesce(`rs`.`expert_review_count`,0) AS `review_count`,coalesce(`rs`.`public_review_count`,0) AS `public_review_count`,coalesce(`rs`.`expert_review_count`,0) AS `expert_review_count`,coalesce(`rps`.`total_reply_count`,0) AS `total_reply_count`,coalesce(`rps`.`total_positive_reactions`,0) AS `total_positive_reply_count`,coalesce(`rps`.`total_negative_reactions`,0) AS `total_negative_reply_count`,coalesce(`rps`.`weekly_total_reply_count`,0) AS `weekly_reply_count`,coalesce(`rps`.`weekly_positive_reply_count`,0) AS `weekly_positive_reply_count`,coalesce(`rps`.`weekly_negative_reply_count`,0) AS `weekly_negative_reply_count` from ((((`on_air_weeks` `oaw` join `movies` `m` on(`oaw`.`movie_id` = `m`.`id`)) left join `box_office` `bo` on(`oaw`.`id` = `bo`.`on_air_week_id`)) left join `review_stats` `rs` on(`oaw`.`id` = `rs`.`week_id`)) left join `reply_stats` `rps` on(`oaw`.`id` = `rps`.`week_id`));

-- ----------------------------
-- View structure for weekly_reviews
-- ----------------------------
DROP VIEW IF EXISTS `weekly_reviews`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `weekly_reviews` AS select `oaw`.`id` AS `on_air_week_id`,`oaw`.`movie_id` AS `movie_id`,`oaw`.`start_date` AS `week_start_date`,`r`.`id` AS `review_id`,`r`.`type` AS `review_type`,`r`.`sentiment_score` AS `sentiment_score`,`r`.`expert_score` AS `expert_score`,char_length(`r`.`content`) AS `content_length`,char_length(`r`.`title`) AS `title_length`,`r`.`created_at` AS `review_created_at` from (`on_air_weeks` `oaw` join `reviews` `r` on(`r`.`movie_id` = `oaw`.`movie_id` and `r`.`created_at` >= `oaw`.`start_date` and `r`.`created_at` < `oaw`.`start_date` + interval 7 day));

-- ----------------------------
-- View structure for zc_input_23_to_25_all_rp
-- ----------------------------
DROP VIEW IF EXISTS `zc_input_23_to_25_all_rp`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `zc_input_23_to_25_all_rp` AS with TargetMovies as (select `on_air_weeks`.`movie_id` AS `movie_id` from `on_air_weeks` group by `on_air_weeks`.`movie_id` having min(`on_air_weeks`.`start_date`) between '2023-01-01' and '2025-12-31'), ReplyStats as (select `replies`.`review_id` AS `review_id`,count(0) AS `total_count`,sum(case when `replies`.`type` = 'push' then 1 else 0 end) AS `positive_count`,sum(case when `replies`.`type` = 'boo' then 1 else 0 end) AS `negative_count` from `replies` group by `replies`.`review_id`)select `r`.`id` AS `review_id`,`r`.`movie_id` AS `movie_id`,`m`.`name` AS `movie_name`,`r`.`title` AS `review_title`,`r`.`url` AS `review_url`,`r`.`content` AS `review_content`,`r`.`created_at` AS `review_date`,`oaw_w0`.`start_date` AS `week_start_date`,`oaw_w0`.`start_date` + interval 6 day AS `week_end_date`,`bo_w0`.`amount` AS `box_office`,`bo_w1`.`amount` AS `box_office_last_week`,`bo_w2`.`amount` AS `box_office_prev_week`,cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) AS `weekly_difference`,cast(`bo_w1`.`amount` as signed) - cast(`bo_w2`.`amount` as signed) AS `prev_weekly_difference`,case when `bo_w1`.`amount` is null or `bo_w1`.`amount` = 0 then NULL else (cast(`bo_w0`.`amount` as double) - `bo_w1`.`amount`) / `bo_w1`.`amount` end AS `percentage_change_last_week`,case when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) > 0 then 1 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) < 0 then 0 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) = 0 then -1 else NULL end AS `weekly_box_office_trend`,`r`.`sentiment_score` AS `sentiment_score`,case when `r`.`sentiment_score` > 0.5 then 1 else 0 end AS `sentiment_score_bool`,coalesce(`rs`.`total_count`,0) AS `reply_count`,coalesce(`rs`.`positive_count`,0) AS `positive_reply_count`,coalesce(`rs`.`negative_count`,0) AS `negative_reply_count` from (((((((((`targetmovies` `tm` join `movies` `m` on(`tm`.`movie_id` = `m`.`id`)) join `reviews` `r` on(`m`.`id` = `r`.`movie_id`)) left join `replystats` `rs` on(`r`.`id` = `rs`.`review_id`)) left join `on_air_weeks` `oaw_w0` on(`r`.`movie_id` = `oaw_w0`.`movie_id` and `r`.`created_at` >= `oaw_w0`.`start_date` and `r`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) join `box_office` `bo_w0` on(`oaw_w0`.`id` = `bo_w0`.`on_air_week_id` and `bo_w0`.`amount` > 0)) left join `on_air_weeks` `oaw_w1` on(`r`.`movie_id` = `oaw_w1`.`movie_id` and `oaw_w1`.`start_date` = `oaw_w0`.`start_date` - interval 7 day)) left join `box_office` `bo_w1` on(`oaw_w1`.`id` = `bo_w1`.`on_air_week_id`)) left join `on_air_weeks` `oaw_w2` on(`r`.`movie_id` = `oaw_w2`.`movie_id` and `oaw_w2`.`start_date` = `oaw_w0`.`start_date` - interval 14 day)) left join `box_office` `bo_w2` on(`oaw_w2`.`id` = `bo_w2`.`on_air_week_id`));

-- ----------------------------
-- View structure for zc_input_23_to_25_filtered_rp
-- ----------------------------
DROP VIEW IF EXISTS `zc_input_23_to_25_filtered_rp`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `zc_input_23_to_25_filtered_rp` AS with TargetMovies as (select `on_air_weeks`.`movie_id` AS `movie_id` from `on_air_weeks` group by `on_air_weeks`.`movie_id` having min(`on_air_weeks`.`start_date`) between '2023-01-01' and '2025-12-31')select `r`.`id` AS `review_id`,`r`.`movie_id` AS `movie_id`,`m`.`name` AS `movie_name`,`r`.`title` AS `review_title`,`r`.`url` AS `review_url`,`r`.`content` AS `review_content`,`r`.`created_at` AS `review_date`,`oaw_w0`.`start_date` AS `week_start_date`,`oaw_w0`.`start_date` + interval 6 day AS `week_end_date`,`bo_w0`.`amount` AS `box_office`,`bo_w1`.`amount` AS `box_office_last_week`,`bo_w2`.`amount` AS `box_office_prev_week`,cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) AS `weekly_difference`,cast(`bo_w1`.`amount` as signed) - cast(`bo_w2`.`amount` as signed) AS `prev_weekly_difference`,case when `bo_w1`.`amount` is null or `bo_w1`.`amount` = 0 then NULL else (cast(`bo_w0`.`amount` as double) - `bo_w1`.`amount`) / `bo_w1`.`amount` end AS `percentage_change_last_week`,case when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) > 0 then 1 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) < 0 then 0 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) = 0 then -1 else NULL end AS `weekly_box_office_trend`,`r`.`sentiment_score` AS `sentiment_score`,case when `r`.`sentiment_score` > 0.5 then 1 else 0 end AS `sentiment_score_bool`,count(`rp`.`id`) AS `reply_count`,sum(case when `rp`.`type` = 'push' then 1 else 0 end) AS `positive_reply_count`,sum(case when `rp`.`type` = 'boo' then 1 else 0 end) AS `negative_reply_count` from (((((((((`targetmovies` `tm` join `movies` `m` on(`tm`.`movie_id` = `m`.`id`)) join `reviews` `r` on(`m`.`id` = `r`.`movie_id`)) left join `on_air_weeks` `oaw_w0` on(`r`.`movie_id` = `oaw_w0`.`movie_id` and `r`.`created_at` >= `oaw_w0`.`start_date` and `r`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) join `box_office` `bo_w0` on(`oaw_w0`.`id` = `bo_w0`.`on_air_week_id` and `bo_w0`.`amount` > 0)) left join `on_air_weeks` `oaw_w1` on(`r`.`movie_id` = `oaw_w1`.`movie_id` and `oaw_w1`.`start_date` = `oaw_w0`.`start_date` - interval 7 day)) left join `box_office` `bo_w1` on(`oaw_w1`.`id` = `bo_w1`.`on_air_week_id`)) left join `on_air_weeks` `oaw_w2` on(`r`.`movie_id` = `oaw_w2`.`movie_id` and `oaw_w2`.`start_date` = `oaw_w0`.`start_date` - interval 14 day)) left join `box_office` `bo_w2` on(`oaw_w2`.`id` = `bo_w2`.`on_air_week_id`)) left join `replies` `rp` on(`r`.`id` = `rp`.`review_id` and `oaw_w0`.`id` is not null and `rp`.`created_at` >= `oaw_w0`.`start_date` and `rp`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) group by `r`.`id`;

-- ----------------------------
-- View structure for zc_input_all_all_rp
-- ----------------------------
DROP VIEW IF EXISTS `zc_input_all_all_rp`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `zc_input_all_all_rp` AS select `r`.`id` AS `review_id`,`r`.`movie_id` AS `movie_id`,`m`.`name` AS `movie_name`,`r`.`title` AS `review_title`,`r`.`url` AS `review_url`,`r`.`content` AS `review_content`,`r`.`created_at` AS `review_date`,`oaw_w0`.`start_date` AS `week_start_date`,`oaw_w0`.`start_date` + interval 6 day AS `week_end_date`,`bo_w0`.`amount` AS `box_office`,`bo_w1`.`amount` AS `box_office_last_week`,`bo_w2`.`amount` AS `box_office_prev_week`,cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) AS `weekly_difference`,cast(`bo_w1`.`amount` as signed) - cast(`bo_w2`.`amount` as signed) AS `prev_weekly_difference`,case when `bo_w1`.`amount` is null or `bo_w1`.`amount` = 0 then NULL else (cast(`bo_w0`.`amount` as double) - `bo_w1`.`amount`) / `bo_w1`.`amount` end AS `percentage_change_last_week`,case when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) > 0 then 1 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) < 0 then 0 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) = 0 then -1 else NULL end AS `weekly_box_office_trend`,`r`.`sentiment_score` AS `sentiment_score`,case when `r`.`sentiment_score` > 0.5 then 1 else 0 end AS `sentiment_score_bool`,coalesce(`rs`.`total_count`,0) AS `reply_count`,coalesce(`rs`.`positive_count`,0) AS `positive_reply_count`,coalesce(`rs`.`negative_count`,0) AS `negative_reply_count` from ((((((((`reviews` `r` join `movies` `m` on(`r`.`movie_id` = `m`.`id`)) left join (select `replies`.`review_id` AS `review_id`,count(0) AS `total_count`,sum(case when `replies`.`type` = 'push' then 1 else 0 end) AS `positive_count`,sum(case when `replies`.`type` = 'boo' then 1 else 0 end) AS `negative_count` from `replies` group by `replies`.`review_id`) `rs` on(`r`.`id` = `rs`.`review_id`)) left join `on_air_weeks` `oaw_w0` on(`r`.`movie_id` = `oaw_w0`.`movie_id` and `r`.`created_at` >= `oaw_w0`.`start_date` and `r`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) join `box_office` `bo_w0` on(`oaw_w0`.`id` = `bo_w0`.`on_air_week_id` and `bo_w0`.`amount` > 0)) left join `on_air_weeks` `oaw_w1` on(`r`.`movie_id` = `oaw_w1`.`movie_id` and `oaw_w1`.`start_date` = `oaw_w0`.`start_date` - interval 7 day)) left join `box_office` `bo_w1` on(`oaw_w1`.`id` = `bo_w1`.`on_air_week_id`)) left join `on_air_weeks` `oaw_w2` on(`r`.`movie_id` = `oaw_w2`.`movie_id` and `oaw_w2`.`start_date` = `oaw_w0`.`start_date` - interval 14 day)) left join `box_office` `bo_w2` on(`oaw_w2`.`id` = `bo_w2`.`on_air_week_id`));

-- ----------------------------
-- View structure for zc_input_all_filtered_rp
-- ----------------------------
DROP VIEW IF EXISTS `zc_input_all_filtered_rp`;
CREATE ALGORITHM = UNDEFINED SQL SECURITY DEFINER VIEW `zc_input_all_filtered_rp` AS select `r`.`id` AS `review_id`,`r`.`movie_id` AS `movie_id`,`m`.`name` AS `movie_name`,`r`.`title` AS `review_title`,`r`.`url` AS `review_url`,`r`.`content` AS `review_content`,`r`.`created_at` AS `review_date`,`oaw_w0`.`start_date` AS `week_start_date`,`oaw_w0`.`start_date` + interval 6 day AS `week_end_date`,`bo_w0`.`amount` AS `box_office`,`bo_w1`.`amount` AS `box_office_last_week`,`bo_w2`.`amount` AS `box_office_prev_week`,cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) AS `weekly_difference`,cast(`bo_w1`.`amount` as signed) - cast(`bo_w2`.`amount` as signed) AS `prev_weekly_difference`,case when `bo_w1`.`amount` is null or `bo_w1`.`amount` = 0 then NULL else (cast(`bo_w0`.`amount` as double) - `bo_w1`.`amount`) / `bo_w1`.`amount` end AS `percentage_change_last_week`,case when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) > 0 then 1 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) < 0 then 0 when cast(`bo_w0`.`amount` as signed) - cast(`bo_w1`.`amount` as signed) = 0 then -1 else NULL end AS `weekly_box_office_trend`,`r`.`sentiment_score` AS `sentiment_score`,case when `r`.`sentiment_score` > 0.5 then 1 else 0 end AS `sentiment_score_bool`,count(`rp`.`id`) AS `reply_count`,sum(case when `rp`.`type` = 'push' then 1 else 0 end) AS `positive_reply_count`,sum(case when `rp`.`type` = 'boo' then 1 else 0 end) AS `negative_reply_count` from ((((((((`reviews` `r` join `movies` `m` on(`r`.`movie_id` = `m`.`id`)) left join `on_air_weeks` `oaw_w0` on(`r`.`movie_id` = `oaw_w0`.`movie_id` and `r`.`created_at` >= `oaw_w0`.`start_date` and `r`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) join `box_office` `bo_w0` on(`oaw_w0`.`id` = `bo_w0`.`on_air_week_id` and `bo_w0`.`amount` > 0)) left join `on_air_weeks` `oaw_w1` on(`r`.`movie_id` = `oaw_w1`.`movie_id` and `oaw_w1`.`start_date` = `oaw_w0`.`start_date` - interval 7 day)) left join `box_office` `bo_w1` on(`oaw_w1`.`id` = `bo_w1`.`on_air_week_id`)) left join `on_air_weeks` `oaw_w2` on(`r`.`movie_id` = `oaw_w2`.`movie_id` and `oaw_w2`.`start_date` = `oaw_w0`.`start_date` - interval 14 day)) left join `box_office` `bo_w2` on(`oaw_w2`.`id` = `bo_w2`.`on_air_week_id`)) left join `replies` `rp` on(`r`.`id` = `rp`.`review_id` and `oaw_w0`.`id` is not null and `rp`.`created_at` >= `oaw_w0`.`start_date` and `rp`.`created_at` < `oaw_w0`.`start_date` + interval 7 day)) group by `r`.`id`;

-- ----------------------------
-- Procedure structure for select_23_to_25_all
-- ----------------------------
DROP PROCEDURE IF EXISTS `select_23_to_25_all`;
delimiter ;;
CREATE PROCEDURE `select_23_to_25_all`()
BEGIN
SELECT * FROM `movie_data_all`.`zc_input_23_to_25_all_rp`;
END
;;
delimiter ;

-- ----------------------------
-- Procedure structure for select_23_to_25_filtered
-- ----------------------------
DROP PROCEDURE IF EXISTS `select_23_to_25_filtered`;
delimiter ;;
CREATE PROCEDURE `select_23_to_25_filtered`()
BEGIN
SELECT * FROM zc_input_23_to_25_filtered_rp;
END
;;
delimiter ;

SET FOREIGN_KEY_CHECKS = 1;
