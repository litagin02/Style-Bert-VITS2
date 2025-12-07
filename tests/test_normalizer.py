from style_bert_vits2.nlp.japanese.normalizer import normalize_text


def test_normalize_text_basic():
    """基本的な正規化のテスト"""
    # 基本的な句読点の正規化
    assert normalize_text("こんにちは。さようなら。") == "こんにちは.さようなら."
    assert normalize_text("おはよう、こんばんは、") == "おはよう,こんばんは,"
    assert normalize_text("すごい！やばい！") == "すごい!やばい!"
    assert normalize_text("なに？どうして？") == "なに?どうして?"
    # 特殊な空白文字
    assert normalize_text("text\u200btext") == "テキストテキスト"  # ゼロ幅スペース
    assert normalize_text("text\u3000text") == "テキスト.テキスト"  # 全角スペース
    assert normalize_text("text\ttext") == "テキストテキスト"  # タブ
    # 制御文字
    assert normalize_text("text\ntext") == "テキスト.テキスト"  # 改行
    assert normalize_text("text\rtext") == "テキストテキスト"  # キャリッジリターン
    # 重複する記号
    assert normalize_text("！！！？？？。。。") == "!!!???..."
    assert normalize_text("。。。、、、") == "...,,,"


def test_normalize_text_units():
    """単位関連の正規化のテスト"""
    # 基本的な単位
    assert normalize_text("100m") == "100メートル"
    assert normalize_text("100cm") == "100センチメートル"
    assert normalize_text("1000.19mm") == "1000.19ミリメートル"
    assert normalize_text("1km") == "1キロメートル"
    assert normalize_text("500mL") == "500ミリリットル"
    assert normalize_text("1L") == "1リットル"
    assert normalize_text("1000.19kL") == "1000.19キロリットル"
    assert normalize_text("1000.19mg") == "1000.19ミリグラム"
    assert normalize_text("100g") == "100グラム"
    assert normalize_text("2kg") == "2キログラム"
    assert normalize_text("200ｍｇ") == "200ミリグラム"  # 全角英数
    # データ容量
    assert normalize_text("51B") == "51バイト"
    assert normalize_text("51KB") == "51キロバイト"
    assert normalize_text("51MB") == "51メガバイト"
    assert normalize_text("51GB") == "51ギガバイト"
    assert normalize_text("51TB") == "51テラバイト"
    assert normalize_text("51PB") == "51ペタバイト"
    assert normalize_text("51EB") == "51エクサバイト"
    assert normalize_text("100KiB") == "100キビバイト"
    assert normalize_text("1000.11MiB") == "1000.11メビバイト"
    assert normalize_text("1000.11GiB") == "1000.11ギビバイト"
    assert normalize_text("1000.11TiB") == "1000.11テビバイト"
    assert normalize_text("1000.11PiB") == "1000.11ペビバイト"
    assert normalize_text("1000.11EiB") == "1000.11エクスビバイト"
    # 面積・体積
    assert normalize_text("100m2") == "100平方メートル"
    assert normalize_text("1km2") == "1平方キロメートル"
    assert normalize_text("50m3") == "50立方メートル"
    # 単位付き指数
    assert normalize_text("1.23e-6") == "零点零零零零零一二三"
    assert normalize_text("1.23e+4") == "一万二千三百"
    assert normalize_text("1.23e-4") == "零点零零零一二三"
    assert normalize_text("1e6") == "百万"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"
    assert normalize_text("100dL〜200dL") == "100デシリットルから200デシリットル"
    # ヘルツ
    assert normalize_text("100Hz") == "100ヘルツ"
    assert normalize_text("100kHz") == "100キロヘルツ"  # k が小文字
    assert normalize_text("100KHz") == "100キロヘルツ"  # K が大文字
    assert normalize_text("100MHz") == "100メガヘルツ"
    assert normalize_text("100GHz") == "100ギガヘルツ"
    assert normalize_text("100THz") == "100テラヘルツ"
    assert normalize_text("45.56kHz") == "45.56キロヘルツ"
    # ヘルツ (hz が小文字、表記揺れ対策)
    assert normalize_text("100hz") == "100ヘルツ"
    assert normalize_text("100khz") == "100キロヘルツ"
    assert normalize_text("100Khz") == "100キロヘルツ"
    assert normalize_text("100Mhz") == "100メガヘルツ"
    assert normalize_text("100Ghz") == "100ギガヘルツ"
    assert normalize_text("100Thz") == "100テラヘルツ"
    assert normalize_text("45.56khz") == "45.56キロヘルツ"
    # ヘクトパスカル
    assert normalize_text("100hPa") == "100ヘクトパスカル"
    assert normalize_text("100hpa") == "100ヘクトパスカル"
    assert normalize_text("100HPa") == "100ヘクトパスカル"
    # アンペア
    assert normalize_text("100A") == "100アンペア"
    assert normalize_text("100mA") == "100ミリアンペア"
    assert normalize_text("100kA") == "100キロアンペア"
    assert normalize_text("45.56mA") == "45.56ミリアンペア"
    # bps
    assert normalize_text("100.55bps") == "100.55ビーピーエス"
    assert normalize_text("100kbps") == "100キロビーピーエス"
    assert normalize_text("100Mbps") == "100メガビーピーエス"
    assert normalize_text("100Gbps") == "100ギガビーピーエス"
    assert normalize_text("100Tbps") == "100テラビーピーエス"
    assert normalize_text("100Pbps") == "100ペタビーピーエス"
    assert normalize_text("100Ebps") == "100エクサビーピーエス"
    # ビット
    assert normalize_text("100bit") == "100ビット"
    assert normalize_text("100kbit") == "100キロビット"
    assert normalize_text("100Mbit") == "100メガビット"
    assert normalize_text("100Gbit") == "100ギガビット"
    assert normalize_text("100Tbit") == "100テラビット"
    assert normalize_text("100Pbit") == "100ペタビット"
    assert normalize_text("100Ebit") == "100エクサビット"
    # スラッシュ付き単位（毎分・毎秒）
    assert normalize_text("100m/h") == "100メートル毎時"
    assert normalize_text("100km/h") == "100キロメートル毎時"
    assert normalize_text("5000m/h") == "5000メートル毎時"
    assert normalize_text("3.5km/h") == "3.5キロメートル毎時"
    assert normalize_text("30.56B/h") == "30.56バイト毎時"
    assert normalize_text("30.56kB/h") == "30.56キロバイト毎時"
    assert normalize_text("30.56KB/h") == "30.56キロバイト毎時"
    assert normalize_text("30.56MB/h") == "30.56メガバイト毎時"
    assert normalize_text("30.56GB/h") == "30.56ギガバイト毎時"
    assert normalize_text("30.56TB/h") == "30.56テラバイト毎時"
    assert normalize_text("30.56EB/h") == "30.56エクサバイト毎時"
    assert normalize_text("30.56b/h") == "30.56ビット毎時"
    assert normalize_text("30.56Kb/h") == "30.56キロビット毎時"
    assert normalize_text("30.56Mb/h") == "30.56メガビット毎時"
    assert normalize_text("30.56Gb/h") == "30.56ギガビット毎時"
    assert normalize_text("30.56Tb/h") == "30.56テラビット毎時"
    assert normalize_text("30.56Eb/h") == "30.56エクサビット毎時"
    assert normalize_text("100m/s") == "100メートル毎秒"
    assert normalize_text("100km/h") == "100キロメートル毎時"
    assert normalize_text("5000m/s") == "5000メートル毎秒"
    assert normalize_text("3.5km/s") == "3.5キロメートル毎秒"
    assert normalize_text("30.56B/s") == "30.56バイト毎秒"
    assert normalize_text("30.56kB/s") == "30.56キロバイト毎秒"
    assert normalize_text("30.56KB/s") == "30.56キロバイト毎秒"
    assert normalize_text("30.56MB/s") == "30.56メガバイト毎秒"
    assert normalize_text("30.56GB/s") == "30.56ギガバイト毎秒"
    assert normalize_text("30.56TB/s") == "30.56テラバイト毎秒"
    assert normalize_text("30.56EB/s") == "30.56エクサバイト毎秒"
    assert normalize_text("30.56b/s") == "30.56ビット毎秒"
    assert normalize_text("30.56Kb/s") == "30.56キロビット毎秒"
    assert normalize_text("30.56Mb/s") == "30.56メガビット毎秒"
    assert normalize_text("30.56Gb/s") == "30.56ギガビット毎秒"
    assert normalize_text("30.56Tb/s") == "30.56テラビット毎秒"
    assert normalize_text("30.56Eb/s") == "30.56エクサビット毎秒"
    # スラッシュ付き単位（毎分・毎秒以外の意図的に変換せず pyopenjtalk に任せるパターン）
    assert normalize_text("100kL/m") == "100kL/m"
    assert normalize_text("100g/㎥") == "100g/m3"
    # スラッシュ付き単位ではないので通常通り変換するパターン (dB は変換対象外の単位)
    assert normalize_text("100m/100.50mL/50dB") == "100メートル/100.50ミリリットル/50dB"
    assert normalize_text("100m/秒") == "100メートル/秒"
    # 英単語の後に単位が来るケース
    assert normalize_text("up to 8GB") == "アップトゥー8ギガバイト"
    # 追加のテストケース
    assert normalize_text("100tトラック") == "100トントラック"
    assert normalize_text("100.1919tトラック") == "100.1919トントラック"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56test") == "345.56テスト"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56ms") == "345.56ミリ秒"
    assert normalize_text("345ms") == "345ミリ秒"
    assert normalize_text("24hを") == "24時間を"
    assert normalize_text("24h営業") == "24時間営業"
    assert normalize_text("24ms営業") == "24ミリ秒営業"
    assert normalize_text("24s営業") == "24秒営業"
    assert normalize_text("500Kがある") == "500Kがある"
    assert normalize_text("50℃") == "50度"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50ns") == "50ナノ秒"
    assert normalize_text("50μs") == "50マイクロ秒"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50h") == "50時間"
    assert normalize_text("1h") == "1時間"
    assert normalize_text("1h3m5s") == "1時間3メートル5秒"
    assert normalize_text("1h5s") == "1時間5秒"
    assert normalize_text("300s") == "300秒"
    assert normalize_text("3hで") == "3時間で"
    assert normalize_text("3hzで") == "3ヘルツで"
    assert normalize_text("30d") == "30日"
    assert normalize_text("30dでなんとかした") == "30日でなんとかした"
    assert normalize_text("でも30dでなんとかした") == "でも30日でなんとかした"
    assert normalize_text("でも30daysでなんとかした") == "でも30デイズでなんとかした"
    assert normalize_text("でも30date") == "でも30デート"
    assert normalize_text("でも30dで") == "でも30日で"
    assert normalize_text("でも30Dで") == "でも30Dで"
    assert normalize_text("\\100") == "100円"
    assert normalize_text("$100で") == "100ドルで"
    assert normalize_text("€100で") == "100ユーロで"
    assert normalize_text("5㎞") == "5キロメートル"
    assert normalize_text("5㎡") == "5平方メートル"


def test_normalize_text_currency():
    """通貨関連の正規化のテスト"""
    # 各種通貨記号
    assert normalize_text("$100") == "100ドル"
    assert normalize_text("¥100") == "100円"
    assert normalize_text("€100") == "100ユーロ"
    assert normalize_text("£100") == "100ポンド"
    assert normalize_text("₩1000") == "1000ウォン"
    # 通貨記号の位置による違い
    assert normalize_text("100$") == "100ドル"
    assert normalize_text("100¥") == "100円"
    # 金額の桁区切り
    assert normalize_text("¥1,234,567") == "1234567円"
    assert normalize_text("$1,234.56") == "1234.56ドル"
    # 通貨の単位
    assert normalize_text("1億円") == "1億円"
    assert normalize_text("100万ドル") == "100万ドル"
    # 特殊な通貨
    assert normalize_text("₿1.5") == "1.5ビットコイン"
    assert normalize_text("₹100") == "100ルピー"
    assert normalize_text("₽50") == "50ルーブル"
    assert normalize_text("₺25") == "25リラ"
    assert normalize_text("฿1000") == "1000バーツ"
    assert normalize_text("₱100") == "100ペソ"
    assert normalize_text("₴50") == "50フリヴニャ"
    assert normalize_text("₫1000") == "1000ドン"
    assert normalize_text("₪100") == "100シェケル"
    assert normalize_text("₦500") == "500ナイラ"
    assert normalize_text("₡1000") == "1000コロン"


def test_normalize_text_dates():
    """日付関連の正規化のテスト"""
    # 様々な日付形式
    assert normalize_text("2024/01/01") == "2024年1月1日"
    assert normalize_text("2024-01-01") == "2024年1月1日"
    assert normalize_text("2024年01月01日") == "2024年1月1日"  # 0埋めを除去
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("1/1") == "1月1日"
    assert normalize_text("２０２４／０１／０１") == "2024年1月1日"  # 全角英数記号
    assert normalize_text("２０２４/０１/０１") == "2024年1月1日"  # 全角英数
    # 曜日付きの日付
    assert normalize_text("2024/01/01(月)") == "2024年1月1日月曜日"
    assert normalize_text("2024-01-01（火）") == "2024年1月1日火曜日"
    assert normalize_text("2024/01/01（月曜）") == "2024年1月1日'月曜'"
    assert normalize_text("2024/01/01（月曜日）") == "2024年1月1日'月曜日'"
    # 年月のみ
    assert normalize_text("2024/01") == "2024年1月"
    assert normalize_text("1930/9") == "1930年9月"
    assert normalize_text("1880/10") == "1880年10月"
    assert normalize_text("2081/12") == "2081年12月"
    assert normalize_text("2181/12") == "2181年12月"
    # 2桁年の自動補完 (50以上は1900年代、49以下は2000年代)
    assert normalize_text("98/01/01") == "1998年1月1日"
    assert normalize_text("24/01/01") == "2024年1月1日"
    # 和暦
    assert normalize_text("令和6年1月1日") == "令和6年1月1日"
    assert normalize_text("平成30年12月31日") == "平成30年12月31日"
    # 日付範囲
    assert normalize_text("1/1〜1/3") == "1月1日から1月3日"
    # 年月
    assert normalize_text("2024年1月") == "2024年1月"
    # 区切り文字のバリエーション
    assert normalize_text("2024.01.01") == "2024年1月1日"
    assert normalize_text("20240101") == "2024年1月1日"
    assert normalize_text("19640820") == "1964年8月20日"
    # 省略表記の和暦
    assert normalize_text("R6.1.1") == "令和6年1月1日"
    assert normalize_text("R6.01.01") == "令和6年1月1日"
    assert normalize_text("H31.4.30") == "平成31年4月30日"
    assert normalize_text("H31.04.30") == "平成31年4月30日"
    assert normalize_text("S64.1.7") == "昭和64年1月7日"
    assert normalize_text("S64.01.07") == "昭和64年1月7日"
    assert normalize_text("S47.12.31") == "昭和47年12月31日"
    # 零時チェック
    assert normalize_text("午前00時") == "午前零時"
    assert normalize_text("午後00時") == "午後零時"
    assert normalize_text("午前00時00分") == "午前零時"
    assert normalize_text("午後00時00分") == "午後零時"
    assert normalize_text("午前00時00分00秒") == "午前零時零分零秒"
    assert normalize_text("午後00時00分00秒") == "午後零時零分零秒"
    assert normalize_text("今日は0時に就寝します") == "今日は零時に就寝します"
    assert normalize_text("今日は00時に就寝します") == "今日は零時に就寝します"
    assert (
        normalize_text("今日は0時間勉強した") == "今日は0時間勉強した"
    )  # 変換されない
    assert normalize_text("1000時間勉強した") == "1000時間勉強した"  # 変換されない
    # 異常な日付
    assert (
        normalize_text("2024/13/01") == "十三ぶんの二千二十四/01"
    )  # 13月は異常値なので分数判定される
    assert (
        normalize_text("2024/01/32") == "2024年1月/32"
    )  # 32日は異常値なので年と月だけ変換される
    assert normalize_text("2024/02/30") == "2024年2月/30"  # 存在しない日付
    assert normalize_text("2024/00/00") == "零ぶんの二千二十四/00"  # ゼロの月日

    # 追加のテストケース
    assert normalize_text("2024年5月8日 （月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年05月08日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("08日　　（月）") == "8日月曜日"
    assert normalize_text("05/31　　（月）") == "5月31日月曜日"
    assert normalize_text("05/30　　（月）") == "5月30日月曜日"
    assert normalize_text("05/20　　（月）") == "5月20日月曜日"
    assert normalize_text("02/21　　（月）") == "2月21日月曜日"
    assert normalize_text("12/21　　（月）") == "12月21日月曜日"
    assert normalize_text("24/12/21　　（月）") == "2024年12月21日月曜日"
    assert normalize_text("24/02/21　　（月）") == "2024年2月21日月曜日"
    assert normalize_text("24/02/1　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01(月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (火)") == "2024年2月1日火曜日"
    assert normalize_text("24/02/01 (水)") == "2024年2月1日水曜日"
    assert normalize_text("24/02/29 (木)") == "2024年2月29日木曜日"
    assert normalize_text("24/02/29 (金)") == "2024年2月29日金曜日"
    assert normalize_text("24/02/29 (土)") == "2024年2月29日土曜日"
    assert normalize_text("24/02/29 （日）") == "2024年2月29日日曜日"
    assert normalize_text("24/02/29") == "2024年2月29日"
    assert normalize_text("98/02/21（水）") == "1998年2月21日水曜日"
    assert normalize_text("98/02/21") == "1998年2月21日"
    assert normalize_text("98/02（水）") == "二ぶんの九十八水曜日"
    assert normalize_text("（水）") == "'水'"
    assert normalize_text("そうだ（水）") == "そうだ'水'"
    assert normalize_text("そうだ（水）に（）行こう") == "そうだ'水'に''行こう"
    assert normalize_text("そうだ（水）に行こう") == "そうだ'水'に行こう"
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("01月03") == "1月03"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("1/3") == "1月3日"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("95/01/03") == "1995年1月3日"
    assert normalize_text("01/03") == "1月3日"
    assert normalize_text("今年01/03にですね") == "今年1月3日にですね"
    assert normalize_text("今年12/3にですね") == "今年12月3日にですね"
    assert normalize_text("今年1/3にですね") == "今年1月3日にですね"
    assert normalize_text("今年9/13にですね") == "今年9月13日にですね"
    assert normalize_text("今年08-13にですね") == "今年08-13にですね"
    assert normalize_text("今年24-08-13にですね") == "今年2024年8月13日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("今年24/12/03にですね") == "今年2024年12月3日にですね"
    assert normalize_text("今年12/03にですね") == "今年12月3日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("20年には") == "20年には"
    assert normalize_text("05年には") == "05年には"
    assert normalize_text("85年には") == "85年には"
    assert normalize_text("05年01月には") == "05年1月には"
    assert normalize_text("09-01-03 24:34") == "2009年1月3日二十四時三十四分"
    assert normalize_text("87-01-03 24:34") == "1987年1月3日二十四時三十四分"
    assert normalize_text("明治45年07月30日") == "明治45年7月30日"
    assert normalize_text("大正15年12月25日") == "大正15年12月25日"
    assert normalize_text("昭和64年01月07日") == "昭和64年1月7日"
    assert normalize_text("平成31年04月30日") == "平成31年4月30日"
    assert normalize_text("令和05年12月31日") == "令和05年12月31日"
    assert normalize_text("西暦2024年1月1日") == "西暦2024年1月1日"
    assert normalize_text("AD2024") == "エーディー2024"
    assert normalize_text("BC356") == "ビーシー356"


def test_normalize_text_time():
    """時刻関連の正規化のテスト"""
    # 基本的な時刻表現
    assert normalize_text("9時30分") == "九時三十分"
    assert normalize_text("14時00分") == "十四時"
    assert normalize_text("7時45分30秒") == "七時四十五分三十秒"
    # コロン区切りの時刻
    assert normalize_text("09:30") == "九時三十分"
    assert normalize_text("14:00") == "十四時"
    assert normalize_text("07:45:30") == "七時四十五分三十秒"
    # アスペクト比（時刻として解釈されない数値の組み合わせ）
    assert normalize_text("16:9") == "十六タイ九"
    # 午前・午後
    assert normalize_text("午前9時30分") == "午前九時三十分"
    assert normalize_text("午後3時45分") == "午後三時四十五分"
    # 特殊な時刻
    assert normalize_text("0時0分") == "零時"
    assert normalize_text("24時00分") == "二十四時"
    assert normalize_text("25:00") == "二十五時"  # 25時までは許容
    assert normalize_text("30:00") == "三十タイ零"  # 30時はアスペクト比として解釈される
    # 秒以下の単位
    assert normalize_text("10時20分30.5秒") == "十時二十分30.5秒"
    # 異常な時刻
    assert normalize_text("24:60") == "二十四時六十"  # 存在しない分
    assert normalize_text("00:00:60") == "零時零分六十"  # 存在しない秒
    # 27時台までは許容
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert (
        normalize_text("28:00:00") == "二十八タイ零タイ零"
    )  # 28時はアスペクト比として解釈される

    # 追加のテストケース
    assert normalize_text("03:34に") == "三時三十四分に"
    assert normalize_text("03:3:564に") == "三タイ三タイ五百六十四に"
    assert normalize_text("03:34:54に") == "三時三十四分五十四秒に"
    assert normalize_text("03:03:03に") == "三時三分三秒に"
    assert normalize_text("03:03:62に") == "三時三分六十二に"
    assert normalize_text("03:03:60に") == "三時三分六十に"
    assert normalize_text("03:03:59に") == "三時三分五十九秒に"
    assert normalize_text("03:03:01に") == "三時三分一秒に"
    assert normalize_text("03:03:5に") == "三時三分五秒に"
    assert normalize_text("04:03") == "四時三分"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("16:3") == "十六タイ三"
    assert normalize_text("04:3") == "四タイ三"
    assert normalize_text("4:30") == "四時三十分"
    assert normalize_text("04:30") == "四時三十分"
    assert normalize_text("04時30分") == "四時三十分"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時六分八秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024:05:08 00:00:00") == "二千二十四タイ五タイ八零時零分零秒"
    assert normalize_text("2024/05/08 00:03:00") == "2024年5月8日零時三分零秒"
    assert normalize_text("2024/05/08 00:03") == "2024年5月8日零時三分"
    assert normalize_text("2024/05/08 00") == "2024年5月8日00"
    assert normalize_text("2024/05/08 0:30") == "2024年5月8日零時三十分"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024/05/01") == "2024年5月1日"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024年05月01日 03時00分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月01日 03時0分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時六分八秒"
    assert normalize_text("2024年05月08日 00時00分30秒") == "2024年5月8日零時零分三十秒"
    assert normalize_text("2024年05月08日 00時00分０0秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分") == "2024年5月8日零時"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時00分0秒") == "2024年5月8日三時零分零秒"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時01分") == "2024年5月8日三時一分"
    assert normalize_text("2024年05月08日 03:01") == "2024年5月8日三時一分"
    assert normalize_text("2024/05/08 03:01") == "2024年5月8日三時一分"
    assert normalize_text("2024/05/08 03:01:00") == "2024年5月8日三時一分零秒"
    assert normalize_text("2024/05/08 3時1分00") == "2024年5月8日三時一分00"
    assert normalize_text("2024/05/08 3時1分0秒") == "2024年5月8日三時一分零秒"
    assert normalize_text("2024/05/08 3時1分60秒") == "2024年5月8日三時一分六十"
    assert normalize_text("2024/05/08 3時1分59秒") == "2024年5月8日三時一分五十九秒"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("27:59") == "二十七時五十九分"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("28:59:00") == "二十八タイ五十九タイ零"
    assert normalize_text("28:59") == "二十八タイ五十九"
    assert normalize_text("03:03:03") == "三時三分三秒"
    assert normalize_text("30:1:03") == "三十タイ一タイ三"
    assert normalize_text("30:1:03:5") == "三十タイ一タイ三,5"
    assert normalize_text("30:1:03:5:07") == "三十タイ一タイ三,五時七分"
    assert normalize_text("1:3:4") == "一タイ三タイ四"
    assert normalize_text("1:3:4:5") == "一タイ三タイ四,5"
    assert normalize_text("1:3:4:5:6") == "一タイ三タイ四,五タイ六"
    assert normalize_text("1:3:4:5:6:7") == "一タイ三タイ四,五タイ六タイ七"
    assert normalize_text("1:3:4:5:6:7:8:9") == "一タイ三タイ四,五タイ六タイ七,八タイ九"
    assert normalize_text("16:9") == "十六タイ九"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("3:2") == "三タイ二"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("3:02") == "三時二分"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("03:02") == "三時二分"
    assert normalize_text("03:02:00") == "三時二分零秒"
    assert normalize_text("03:00:00") == "三時零分零秒"
    assert normalize_text("24:00:00") == "二十四時零分零秒"
    assert normalize_text("24:00") == "二十四時"
    assert normalize_text("27:59:00") == "二十七時五十九分零秒"
    assert normalize_text("00:00:00.123") == "零時零分零秒.123"
    assert normalize_text("12:00 PM") == "十二時ピーエム"
    assert normalize_text("12:00 AM") == "十二時エーエム"
    assert normalize_text("深夜03時") == "深夜3時"
    assert normalize_text("未明04時") == "未明4時"
    assert normalize_text("早朝05時") == "早朝5時"
    assert normalize_text("夜09時") == "夜9時"
    assert normalize_text("正午") == "正午"
    assert normalize_text("正午12時") == "正午12時"
    assert normalize_text("0:00:00") == "零時零分零秒"


def test_normalize_text_fractions():
    """分数関連の正規化のテスト (明確に日付ではないパターンのみ分数として読まれる)"""
    assert normalize_text("123/456") == "四百五十六ぶんの百二十三"
    assert normalize_text("1/100") == "百ぶんの一"
    assert normalize_text("13/32") == "三十二ぶんの十三"
    # 分数を含む文
    assert normalize_text("材料の2/30を使用した。") == "材料の三十ぶんの二を使用した."
    assert normalize_text("残り時間は1/40です。") == "残り時間は四十ぶんの一です."

    # 追加のテストケース
    assert normalize_text("16/9") == "九ぶんの十六"
    assert normalize_text("1/2") == "1月2日"  # 日付として解釈される
    assert normalize_text("1/32") == "三十二ぶんの一"
    assert normalize_text("9/16") == "9月16日"  # 日付として解釈される


def test_normalize_text_symbols():
    """記号関連の正規化のテスト"""
    # 基本的な記号
    assert normalize_text("ABC+ABC") == "エービーシープラスエービーシー"
    assert normalize_text("ABC&ABC") == "エービーシーアンドエービーシー"
    assert normalize_text("abc+abc") == "エービーシープラスエービーシー"
    assert normalize_text("abc&abc") == "エービーシーアンドエービーシー"
    assert (
        normalize_text("OpenAPI-Specification")
        == "オープンエーピーアイスペシフィケーション"
    )
    # 数式
    assert normalize_text("1+1=2") == "1プラス1イコール2"
    assert normalize_text("5-3=2") == "5マイナス3イコール2"
    assert normalize_text("2×3=6") == "2かける3イコール6"
    assert normalize_text("6÷2=3") == "6わる2イコール3"
    # 比較演算子
    assert normalize_text("5>3") == "5大なり3"
    assert normalize_text("5≥3") == "5大なりイコール3"
    assert normalize_text("2<4") == "2小なり4"
    assert normalize_text("2≤4") == "2小なりイコール4"


def test_normalize_text_url_email():
    """URL・メールアドレス関連の正規化のテスト"""
    # URL
    assert (
        normalize_text("https://example.com")
        == "エイチティーティーピーエス,イグザンプルドットコム"
    )
    assert (
        normalize_text("http://test.jp")
        == "エイチティーティーピー,テストドットジェイピー"
    )
    # 全角 URL
    assert (
        normalize_text("ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ")
        == "エイチティーティーピーエス,イグザンプルドットコム"
    )
    # メールアドレス
    assert (
        normalize_text("test@example.com")
        == "テスト,アットマーク,イグザンプルドットコム"
    )
    assert (
        normalize_text("info@test.co.jp")
        == "インフォ,アットマーク,テストドットシーオードットジェイピー"
    )


def test_normalize_text_divider_blocks():
    """区切り用途の連続記号を句点に畳み込むテスト"""
    assert normalize_text("###########") == "."
    assert normalize_text("-------------") == "."
    assert normalize_text("_____   _____") == "."
    assert normalize_text(":::::") == "."
    assert normalize_text("*****") == "."
    assert normalize_text("#$#$#$#") == "."
    assert normalize_text("そうなんだよな ##### でもなぁ") == "そうなんだよな.でもなぁ"
    # 複数行
    assert (
        normalize_text("""--------------------------
######### これはコメントです #########
=================
""")
        == ".これはコメントです."
    )
    # このような感情表現としての連続する記号は変換されない
    assert (
        normalize_text(
            "やった〜〜〜〜！！！！テストでようやく満点取れたよ・・・・・・………。。。。。。あなたはどう?????!!!"
        )
        == "やったーーーー!!!!テストでようやく満点取れたよ.....................あなたはどう?????!!!"
    )


def test_normalize_text_english():
    """英語関連の正規化のテスト"""
    # 基本的な英単語
    assert normalize_text("Hello") == "ハロー"
    assert normalize_text("Good Morning") == "グッドモーニング"
    assert normalize_text("Node.js") == "ノードジェイエス"
    # 複数形
    assert normalize_text("computers") == "コンピューターズ"
    assert normalize_text("smartphones") == "スマートフォーンズ"
    assert (
        normalize_text("chatgpts")
        == "チャットジーピーティーズ"  # "chatgpt" しか辞書に含まれていない場合に自動的に s をつけて読む
    )
    # CamelCase
    assert normalize_text("JavaScript") == "ジャバスクリプト"
    assert normalize_text("TypeScript") == "タイプスクリプト"
    assert normalize_text("RockchipTechnologies") == "ロックチップテクノロジーズ"
    assert normalize_text("splitTextAndImage()") == "スプリットテキストアンドイメージ''"
    # 複合語
    assert normalize_text("e-mail") == "イーメール"
    assert normalize_text("YouTube") == "ユーチューブ"
    # 辞書にない単語の変換
    assert (
        # 小文字の場合は2単語へ分割して辞書で変換
        normalize_text("windsurfeditor") == "ウインドサーフエディター"
    )
    assert (
        # 大文字の場合は2単語への分割が行われないので、C2K によるカタカナ読み推定結果が返る
        # Updated to match current e2k git version (commit 9920754e)
        normalize_text("WINDSURFEDITOR") == "ウィンサーフデディター"
    )
    assert normalize_text("DevinProgrammerAgents") == "デビンプログラマーエージェンツ"
    # クオートの正規化処理
    assert normalize_text("I'm") == "アイム"
    assert normalize_text("I’m") == "アイム"
    assert normalize_text("We've") == "ウイブ"
    assert normalize_text("We’ve") == "ウイブ"

    # 敬称の処理
    assert normalize_text("Mr. John") == "ミスタージョン"
    assert normalize_text("Mrs. Smith") == "ミセススミス"
    assert normalize_text("Ms. Jane") == "ミズジェーン"
    assert normalize_text("Dr. Brown") == "ドクターブラウン"
    assert normalize_text("John Smith Jr.") == "ジョンスミスジュニア"
    assert normalize_text("John Smith Sr.") == "ジョンスミスシニア"
    assert normalize_text("Mr Smith") == "ミスタースミス"  # ピリオドなし
    assert normalize_text("Dr Brown") == "ドクターブラウン"  # ピリオドなし
    assert normalize_text("Mr. and Mrs. Smith") == "ミスターアンドミセススミス"
    assert normalize_text("Dr. Smith Jr.") == "ドクタースミスジュニア"
    assert (
        normalize_text("Mr. John Smith Jr. PhD")
        == "ミスタージョンスミスジュニアピーエイチディー"
    )
    assert normalize_text("John's book") == "ジョンズブック"
    assert normalize_text("The company's policy") == "ザカンパニーズポリシー"

    # 英単語の後に数字が来る場合
    assert normalize_text("GPT-3") == "ジーピーティースリー"
    assert normalize_text("GPT-11") == "ジーピーティーイレブン"
    assert (
        # 小数点は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("GPT-4.5") == "ジーピーティー4.5"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("GPT-12") == "ジーピーティー12"
    )
    assert normalize_text("iPhone11") == "アイフォンイレブン"
    assert normalize_text("iPhone 8") == "アイフォンエイト"
    assert normalize_text("iPhone 9 Pro Max") == "アイフォンナインプロマックス"
    assert normalize_text("Claude 3") == "クロードスリー"
    assert normalize_text("Pixel 8") == "ピクセルエイト"
    assert (
        # 09 のように数字が0埋めされている場合は変換しない
        normalize_text("Pixel 09") == "ピクセル09"
    )
    assert (
        # 8a のように数字の後にスペースなしで何か付く場合は変換しない
        normalize_text("Pixel 8a") == "ピクセル8a"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("iPhone12") == "アイフォン12"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Android 14") == "アンドロイド14"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Windows 95") == "ウィンドウズ95"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Windows95") == "ウィンドウズ95"
    )
    assert normalize_text("Gemini-2") == "ジェミニツー"
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text("Gemini-1.5") == "ジェミニ1.5"
    )
    assert normalize_text("Gemini 2") == "ジェミニツー"
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text("Gemini 2.0") == "ジェミニ2.0"
    )
    assert (
        # ハイフンが Non-Breaking Hyphen になっている
        normalize_text("GPT‑4") == "ジーピーティーフォー"
    )

    # 単数を表す "a" の処理
    assert normalize_text("a pen") == "アペン"
    assert normalize_text("a book") == "アブック"
    assert normalize_text("a student") == "アスチューデント"
    assert normalize_text("not a pen") == "ノットアペン"
    assert (
        # OFDMEXA は辞書未収録の造語かつ全て大文字で、NGram によって英単語として読むべきと判定されるのでそのまま
        normalize_text("a OFDMEXA modular") == "アOFDMEXAモジュラー"
    )
    assert normalize_text("a 123") == "a123"  # 数字の前の a はそのまま
    assert normalize_text("This is a pen.") == "ディスイズアペン."
    assert normalize_text("This is a good pen.") == "ディスイズアグッドペン."

    # ハイフンで区切られた英単語の処理
    assert normalize_text("pen") == "ペン"
    assert normalize_text("good-pen") == "グッドペン"
    assert normalize_text("OFDMEXA-modular") == "OFDMEXAモジュラー"
    assert (
        # "Bentol" は適当にでっち上げた造語なので C2K によってカタカナ推定が入り、それ以外は辞書からカタカナ表記が取得される
        normalize_text("Bentol-API-SpecificationResult2")
        == "ベントルエーピーアイスペシフィケーションリザルトツー"
    )
    assert (
        # "Paravoice" は辞書にない造語なので、C2K によってカタカナ推定が入る
        # "demo" は辞書に存在するため、辞書からカタカナ表記が取得される
        # Updated to match current e2k git version (commit 9920754e)
        normalize_text("Paravoice 3を4GBまでdemo試せます")
        == "パラヴォイススリーを4ギガバイトまでデモ試せます"
    )

    # ローマ字読み (辞書に存在するもの)
    assert normalize_text("Akagi") == "アカギ"
    assert normalize_text("Akagisan") == "アカギサン"
    assert normalize_text("Akahata") == "アカハタ"
    assert normalize_text("Akasaka") == "アカサカ"
    assert normalize_text("Akashi") == "アカシ"
    assert normalize_text("Akashiyaki") == "アカシヤキ"
    assert normalize_text("Akebono") == "アケボノ"
    assert normalize_text("Akihabara") == "アキハバラ"
    assert normalize_text("Akita") == "アキタ"
    assert normalize_text("Akitainu") == "アキタイヌ"
    assert normalize_text("Amae") == "アマエ"
    assert normalize_text("Amagasaki") == "アマガサキ"
    assert normalize_text("Amakusa") == "アマクサ"
    assert normalize_text("Amazake") == "アマザケ"
    assert normalize_text("Amezaiku") == "アメザイク"

    # ローマ字読み (辞書にもなく、NGram が英単語ではないと判定するため、そのまま返される)
    # pyopenjtalk 側でアルファベット読み/ローマ字読みされる
    assert (
        # "MIZU" (大文字) は辞書になく、NGram が False を返すため、そのまま "MIZU" として返される
        # その後 pyopenjtalk でローマ字として読まれる想定
        normalize_text("MIZU") == "MIZU"
    )
    assert (
        # "mizu" (小文字) は辞書になく、NGram が False を返すが、大文字ではないため
        # C2K で処理され、カタカナ "ミズ" に変換される
        normalize_text("mizu") == "ミズ"
    )

    # ローマ字読み (C2K でいい感じに自動推定される)
    assert (
        # Updated to match current e2k git version (commit 9920754e)
        normalize_text("KONO DENSHAWA YAMANOTESEN UCHIMAWARI")
        == "コノデンシャワヤマノテセンウチマワリ"
    )
    assert (
        normalize_text("Next Station Is Musashi-Mizonokuchi.")
        == "ネクストステイションイズムサシミゾノクチ."
    )
    assert normalize_text("ConoHa") == "コノハ"
    assert normalize_text("KonoHa") == "コノハ"
    assert normalize_text("QonoHa") == "コノハ"

    # 長い英文
    assert (
        normalize_text(
            "GPT-4 can solve difficult problems with greater accuracy, thanks to its broader general knowledge and problem solving abilities."
        )
        == "ジーピーティーフォーキャンソルブディフィカルトプロブレムズウィズグレーターアキュラシー,サンクストゥーイツブローダージェネラルナレッジアンドプロブレムソルビングアビリティーズ."
    )
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text(
            "We’re releasing a research preview of GPT‑4.5—our largest and best model for chat yet. GPT‑4.5 is a step forward in scaling up pre-training and post-training. By scaling unsupervised learning, GPT‑4.5 improves its ability to recognize patterns, draw connections, and generate creative insights without reasoning."
        )
        == "ウイアーリリーシングアリサーチプレビューオブジーピーティー4.5-アワーラージェストアンドベストモデルフォーチャットイェット.ジーピーティー4.5イズアステップフォーワードインスケーリングアッププリートレーニングアンドポストトレーニング.バイスケーリングアンスーパーバイズドラーニング,ジーピーティー4.5インプルーブズイツアビリティートゥーレコグナイズパターンズ,ドローコネクションズ,アンドジェネレートクリエイティブインサイツウィザウトリーズニング."
    )

    # 複雑な CamelCase と英文の混合パターン (TODO: 改善の余地あり)
    assert (
        normalize_text(
            "ではCinamicさん、WindsurfCascade-PriceはGemini+Claude&Deepseekesより安いか分かりますか？"
        )
        == "ではシナマイクさん,ウインドサーフカスケードプライスはジェミニプラスクロードアンドディープシークエスより安いか分かりますか?"
    )
    assert (
        normalize_text("I'm human, with ApplePencil. Because, We have iPhone 8.")
        == "アイムヒューマン,ウィズアップルペンシル.ビコーズ,ウィーハブアイフォンエイト."
    )
    assert (
        # "GPT" は辞書に登録されているため "ジーピーティー" に変換される
        normalize_text("ModelTrainingWithGPT4TurboAndLlama3")
        == "モデルトレーニングウィズジーピーティー4ターボアンドラマスリー"
    )
    assert (
        normalize_text("NextGenCloudComputingSystem2024")
        == "ネクストジェンクラウドコンピューティングシステム2024"
    )
    assert (
        normalize_text("MachineLearning+DeepLearning=AI")
        == "マシンラーニングプラスディープラーニングイコールエーアイ"
    )
    assert (
        # "iPhone" は辞書に登録されているため "アイフォン" に変換される
        normalize_text("iPhoneProMax15-vs-GooglePixel8 Pro")
        == "アイフォンプロマックス15バーサスグーグルピクセルエイトプロ"
    )
    assert (
        normalize_text("WebDev2023: HTML5+CSS3+JavaScript")
        == "ウェブデブ2023,エイチティーエムエルファイブプラスシーエスエススリープラスジャバスクリプト"
    )


def test_normalize_text_mathematical():
    """数学記号関連の正規化のテスト"""
    # 数学記号
    assert normalize_text("∞") == "無限大"
    assert normalize_text("π") == "パイ"
    assert normalize_text("√4") == "ルート4"
    assert normalize_text("∛8") == "立方根8"
    assert normalize_text("∜16") == "四乗根16"
    assert normalize_text("∑") == "シグマ"
    assert normalize_text("∫") == "インテグラル"
    assert normalize_text("∬") == "二重積分"
    assert normalize_text("∭") == "三重積分"
    assert normalize_text("∮") == "周回積分"
    assert normalize_text("∯") == "面積分"
    assert normalize_text("∰") == "体積分"
    assert normalize_text("∂") == "パーシャル"
    assert normalize_text("∇") == "ナブラ"
    assert normalize_text("∝") == "比例"
    # 集合記号
    assert normalize_text("∈") == "属する"
    assert normalize_text("∉") == "属さない"
    assert normalize_text("∋") == "含む"
    assert normalize_text("∌") == "含まない"
    assert normalize_text("∪") == "和集合"
    assert normalize_text("∩") == "共通部分"
    assert normalize_text("⊂") == "部分集合"
    assert normalize_text("⊃") == "上位集合"
    assert normalize_text("⊄") == "部分集合でない"
    assert normalize_text("⊅") == "上位集合でない"
    assert normalize_text("⊆") == "部分集合または等しい"
    assert normalize_text("⊇") == "上位集合または等しい"
    assert normalize_text("∅") == "空集合"
    assert normalize_text("∖") == "差集合"
    assert normalize_text("∆") == "対称差"
    # 幾何記号
    assert normalize_text("∥") == "平行"
    assert normalize_text("⊥") == "垂直"
    assert normalize_text("∠") == "角"
    assert normalize_text("∟") == "直角"
    assert normalize_text("∡") == "測定角"
    assert normalize_text("∢") == "球面角"


def test_normalize_text_ranges():
    """範囲表現の正規化のテスト"""
    # 数値範囲
    assert normalize_text("1〜10") == "1から10"
    assert normalize_text("1~10") == "1から10"
    assert normalize_text("1～10") == "1から10"
    # 文字を含む範囲
    assert normalize_text("AからZ") == "AからZ"
    assert normalize_text("1から100まで") == "1から100まで"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"


def test_normalize_text_enclosed_characters():
    """囲み文字の正規化のテスト"""
    # 丸付き数字
    assert normalize_text("①②③④⑤⑥⑦⑧⑨⑩") == "12345678910"
    assert normalize_text("⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳") == "11121314151617181920"
    # 囲み文字（漢字）
    assert normalize_text("㈱") == "株式会社"
    assert normalize_text("㈲") == "有限会社"
    assert normalize_text("㈳") == "社団法人"
    assert normalize_text("㈴") == "合名会社"
    assert normalize_text("㈵") == "特殊法人"
    assert normalize_text("㈶") == "財団法人"
    assert normalize_text("㈷") == "祝日"
    assert normalize_text("㈸") == "労働組合"
    assert normalize_text("㈹") == "代表電話"
    assert normalize_text("㈺") == "呼出し電話"
    assert normalize_text("㈻") == "学校法人"
    assert normalize_text("㈼") == "監査法人"
    assert normalize_text("㈽") == "企業組合"
    assert normalize_text("㈾") == "合資会社"
    assert normalize_text("㈿") == "協同組合"
    assert normalize_text("㊤㊥㊦") == "上中下"
    assert normalize_text("㊧㊨") == "左右"
    assert normalize_text("㊩") == "医療法人"
    assert normalize_text("㊪") == "宗教法人"
    assert normalize_text("㊫") == "学校法人"
    assert normalize_text("㊬") == "監査法人"
    assert normalize_text("㊭") == "企業組合"
    assert normalize_text("㊮") == "合資会社"
    assert normalize_text("㊯") == "協同組合"


def test_normalize_text_mixed_scripts():
    """文字種混在のテスト"""
    # 漢字・ひらがな・カタカナの混在
    assert (
        normalize_text("漢字とひらがなとカタカナの混在文")
        == "漢字とひらがなとカタカナの混在文"
    )
    # 英数字との混在
    assert normalize_text("123と漢字とABCの混在") == "123と漢字とエービーシーの混在"
    # 記号との混在
    assert normalize_text("漢字+カタカナ=混在!?") == "漢字プラスカタカナイコール混在!?"
    # 特殊文字との混在
    assert normalize_text("①漢字②ひらがな③カタカナ") == "1漢字2ひらがな3カタカナ"
    # 単位との混在
    assert (
        normalize_text("漢字100kg+カタカナ500m")
        == "漢字100キログラムプラスカタカナ500メートル"
    )


def test_normalize_text_edge_cases():
    """エッジケースの正規化のテスト"""
    # 空文字列
    assert normalize_text("") == ""
    # 記号のみ
    assert normalize_text("...") == "..."
    assert normalize_text("!!!") == "!!!"
    assert normalize_text("???") == "???"
    # 数字のみ
    assert normalize_text("12345") == "12345"

    # 結合文字の濁点・半濁点
    assert normalize_text("か゛") == "か"  # 結合文字の濁点は削除
    assert normalize_text("は゜") == "は"  # 結合文字の半濁点は削除

    # 極端に長い数値
    assert normalize_text("12345678901234567890") == "12345678901234567890"
    # 極端に長い英単語
    assert (
        normalize_text("supercalifragilisticexpialidocious")
        # e2k ライブラリによる自動推定結果
        # Updated to match current e2k git version (commit 9920754e)
        == "スーパーカリフラジリー"
    )
    # 特殊な文字の組み合わせ
    assert normalize_text("㊊㊋㊌㊍㊎㊏㊐") == "月火水木金土日"  # 曜日の丸文字
    assert (
        normalize_text("㍉㌔㌢㍍㌘㌧㌃㌶㍑㍗")
        == "ミリキロセンチメートルグラムトンアールヘクタールリットルワット"
    )


def test_normalize_text_complex():
    """複合的なパターンの正規化のテスト"""
    # 日付・時刻・単位を含む文
    assert (
        normalize_text("2024/01/01(月)の14時30分に1.5kgの荷物を受け取った。")
        == "2024年1月1日月曜日の十四時三十分に1.5キログラムの荷物を受け取った."
    )
    assert (
        normalize_text("MacBookで1080p/60fpsの動画を2GB保存した。")
        == "マックブックで1080p/60エフピーエスの動画を2ギガバイト保存した."
    )
    assert (
        normalize_text("¥1,000の商品を2個買うと、¥2,000です（1,000×2=2,000）。")
        == "1000円の商品を2個買うと,2000円です'1000かける2イコール2000'."
    )
    assert (
        normalize_text(
            "お問い合わせは、info@example.comまたはhttps://example.com/contactまで！"
        )
        == "お問い合わせは,インフォ,アットマーク,イグザンプルドットコムまたはエイチティーティーピーエス,イグザンプルドットコム,スラッシュ,コンタクトまで!"
    )
    assert (
        normalize_text("09:30に家を出発し、2km先のスーパーで500gのお肉を買った。")
        == "九時三十分に家を出発し,2キロメートル先のスーパーで500グラムのお肉を買った."
    )
    assert (
        normalize_text(
            "2024/05/01にWindowsのアップデート（2GB+500MB=2.5GB）を実施する。"
        )
        == "2024年5月1日にウィンドウズのアップデート'2ギガバイトプラス500メガバイトイコール2.5ギガバイト'を実施する."
    )
    assert (
        normalize_text(
            "CPU使用率が50%を超え、メモリ消費が2GBに達した時点で、Windows Serverは自動的に再起動します。"
        )
        == "シーピーユー使用率が50パーセントを超え,メモリ消費が2ギガバイトに達した時点で,ウィンドウズサーバーは自動的に再起動します."
    )
    assert (
        normalize_text(
            "新商品のiPhone 15 Pro Max (256GB)が¥158,000(税込)で発売！9/22(金)午前10時から予約受付開始。"
        )
        == "新商品のアイフォン15プロマックス'256ギガバイト'が158000円'税込'で発売!9月22日金曜日午前10時から予約受付開始."
    )
    assert (
        normalize_text(
            "株式会社Deeptest(担当：山田)様、10/1(月)15:00〜17:00にWeb会議(https://meet.example.com/test)を設定しました。"
        )
        == "株式会社ディープテスト'担当,山田'様,10月1日月曜日十五時から十七時にウェブ会議'エイチティーティーピーエス,ミートドット,イグザンプルドットコム,スラッシュ,テスト'を設定しました."
    )
    assert (
        normalize_text(
            "材料(4人分)：牛肉250g、玉ねぎ1個、水300mL、醤油大さじ2(30mL)、砂糖20g。"
        )
        == "材料'4人分',牛肉250グラム,玉ねぎ1個,水300ミリリットル,醤油大さじ2'30ミリリットル',砂糖20グラム."
    )
    assert (
        normalize_text(
            "2つの数 a, b があり、a:b = 2:3 で、a + b = 10 のとき、a = 4, b = 6 となります。"
        )
        == "2つの数a,bがあり,a,bイコール二タイ三で,aプラスbイコール10のとき,aイコール4,bイコール6となります."
    )
    assert (
        normalize_text(
            "JavaScriptでArray.prototype.map()を使用し、配列の要素を2倍にする処理を1/100秒で実行。"
        )
        == "ジャバスクリプトでアレイプロトタイプマップ''を使用し,配列の要素を2倍にする処理を百ぶんの一秒で実行."
    )
    assert (
        normalize_text(
            "今日01/03（月）にですね、16:9の映像を1/128の確率で表示するイベントをやっていて、85/09/30の08月01日(金)にお会いした人と久々に会うんです"
        )
        == "今日1月3日月曜日にですね,十六タイ九の映像を百二十八ぶんの一の確率で表示するイベントをやっていて,1985年9月30日の8月1日金曜日にお会いした人と久々に会うんです"
    )
    assert (
        normalize_text(
            "path-to-model-file.onnx は事前学習済みの onnx モデルファイルです。 onnx_model/phoneme_transition_model.onnxにあります。 path-to-wav-file はサンプリング周波数 16kHz  のモノラル wav ファイルです。 path-to-phoneme-file は音素を空白区切りしたテキストが格納されたファイルのパスです。 NOTE: 開始音素と終了音素は pau である必要があります。"
        )
        == "パストゥーモデルファイルオニキスは事前学習済みのオニキスモデルファイルです.オニキスモデル/フォーニムトランジションモデルオニキスにあります.パストゥーワブファイルはサンプリング周波数16キロヘルツのモノラルワブファイルです.パストゥーフォーニムファイルは音素を空白区切りしたテキストが格納されたファイルのパスです.ノート,開始音素と終了音素はパウである必要があります."
    )
    assert (
        normalize_text(
            "Apple Watch Series 10も安くなっている。Amazonでの 販売価格は、昨年発売された新モデルということもあり、過去最安値となっている。42mmのジェットブラックモデル（Wi-Fi）の場合、5％OFFの\\53,693＋537ポイントで販売している。ヨドバシ.comとビックカメラ.comでもセール対象となっており、ポイント還元分を含んだ実質価格はAmazonと同等だ。"
        )
        == "アップルウォッチシリーズテンも安くなっている.アマゾンでの販売価格は,昨年発売された新モデルということもあり,過去最安値となっている.42ミリメートルのジェットブラックモデル'ワイファイ'の場合,5パーセントオフの53693円プラス537ポイントで販売している.ヨドバシ.コムとビックカメラ.コムでもセール対象となっており,ポイント還元分を含んだ実質価格はアマゾンと同等だ."
    )
    assert (
        normalize_text(
            "音質面では、8W×2基のスピーカーを搭載。立体音響フォーマットはDTS:Xに対応し、バーチャルサウンド技術のDTS:Virtualサウンドを活用した再生も可能だという。Google TVが導入されているため、YouTube／Prime Video／Netflixといった多数のVODサービスが楽しめる他、音声操作のGoogleアシスタントbuilt-in、スマートフォンなどのデバイスからテレビに映像をキャストするChromecast built-inなども採用されている。ユニボディデザインに極上メタリックフレームを採用したプレミアムなデザインも特徴的。付属リモコンは、Bluetooth接続タイプが投入されている。ワイヤレス機能は、Bluetooth ver5.0、Wi-Fi（5GHz/2.4GHz）に対応する。"
        )
        == "音質面では,8Wかける2基のスピーカーを搭載.立体音響フォーマットはディーティーエス,Xに対応し,バーチャルサウンド技術のディーティーエス,バーチャルサウンドを活用した再生も可能だという.グーグルティービーが導入されているため,ユーチューブ/プライムビデオ/ネットフリックスといった多数のブイーオーディーサービスが楽しめる他,音声操作のグーグルアシスタントビルトイン,スマートフォンなどのデバイスからテレビに映像をキャストするクロームキャストビルトインなども採用されている.ユニボディデザインに極上メタリックフレームを採用したプレミアムなデザインも特徴的.付属リモコンは,ブルートゥース接続タイプが投入されている.ワイヤレス機能は,ブルートゥースバー5.0,ワイファイ'5ギガヘルツ/2.4ギガヘルツ'に対応する."
    )
    assert (
        normalize_text(
            "Scopely（スコープリー）は『モノポリーGO』や『マーベル・ストライクフォース』などを配信している、アメリカのモバイルゲーム会社。2023年にサウジアラビアのSavvy Games Groupに49億ドルで買収されている。 一方のナイアンティックは、前述のゲーム事業の売却に合わせて、新会社となる”Niantic Spatial Inc.”（ナイアンティックスペーシャル）を設立。ジオスペーシャルAI事業として、空間コンピューティング、XR、地理情報システム（GIS）、AIを統合した、新たなプラットフォーム”Niantic Spatial Platform”へ注力するという。なお、『ポケモンGO』や『モンスターハンターNow』、『ピクミンブルーム』の事業はスコープリーへ移管されるものの、『Ingress Prime』や『Peridot』などの現実世界を舞台にしたARゲームは引き続き、ナイアンティック側で運営を行う。"
        )
        == "スコープリー'スコープリー'はモノポリーゴーやマーベル,ストライクフォースなどを配信している,アメリカのモバイルゲーム会社.2023年にサウジアラビアのサヴィゲームズグループに49億ドルで買収されている.一方のナイアンティックは,前述のゲーム事業の売却に合わせて,新会社となる'ナイアンティックスペイシャルインク.''ナイアンティックスペーシャル'を設立.ジオスペーシャルエーアイ事業として,空間コンピューティング,エックスアール,地理情報システム'ジーアイエス',エーアイを統合した,新たなプラットフォーム'ナイアンティックスペイシャルプラットフォーム'へ注力するという.なお,ポケモンゴーやモンスターハンターナウ,ピクミンブルームの事業はスコープリーへ移管されるものの,イングレスプライムやペリドットなどの現実世界を舞台にしたエーアールゲームは引き続き,ナイアンティック側で運営を行う."
    )
    assert (
        normalize_text(
            "ROCK5 is a series of Rockchip RK3588(s) based SBC(Single Board Computer) by Radxa. It can run Linux, Android, BSD and other distributions. ROCK5 comes in two models, Model A and Model B. Both models offer 4GB, 8GB, 16GB and 32GB options. For detailed difference between Model A and Model B, please check Specifications. ROCK5 features a Octa core ARM processor(4x Cortex-A76 + 4x Cortex-A55), 64bit 3200Mb/s LPDDR4, up to 8K@60 HDMI, MIPI DSI, MIPI CSI, 3.5mm jack with mic, USB Port, 2.5 GbE LAN, PCIe 3.0, PCIe 2.0, 40-pin color expansion header, RTC. Also, ROCK5 supports USB PD and QC powering."
        )
        # Updated to match current e2k git version (commit 9920754e)
        == "ロックファイブイズアシリーズオブロックチップRK3588's'ベースドエスビーシー'シングルボードコンピューター'バイラダ.イットキャンランリナックス,アンドロイド,ビーエスディーアンドアザーディストリビューションズ.ロックファイブカムズインツーモデルズ,モデルAアンドモデルB.ボスモデルズオファー4ギガバイト,8ギガバイト,16ギガバイトアンド32ギガバイトオプションズ.フォーディテールズディファレンスビトゥイーンモデルAアンドモデルB,プリーズチェックスペシフィケーションズ.ロックファイブフィーチャーズアオクタコアアームプロセッサー'4xコーテックスA76プラス4xコーテックスA55',64ビット3200メガビット毎秒エルピーディーディーアールフォー,アップトゥーはちケー60エイチディーエムアイ,ミピーディーエスアイ,ミピーシーエスアイ,3.5ミリメートルジャックウィズマイク,ユーエスビーポート,2.5ジービーイーラン,ピーシーアイイー3.0,ピーシーアイイー2.0,40ピンカラーエクスパンションヘッダー,アールティーシー.オルソ,ロックファイブサポーツユーエスビーピーディーアンドキューシーパワーリング."
    )
