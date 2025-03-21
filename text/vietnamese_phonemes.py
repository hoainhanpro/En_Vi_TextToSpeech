"""
    Created by Son Pham
    Email: sp@outlook.com.vn
    Licsence: MIT license
"""

valid_symbols = ["a", "ai", "am", "an", "ang", "anh", "ao", "au", "b", "ay", "e", "em", "en", "eng", "eo", "i", "ia", "in", "inh", "ieem", "ieen", "ieec1", "ieem1", "ieen1", "ieeng1", "ieet1", "ieeu1", "ieen2", "ieeu2", "ieen3", "ieeu3", "ieeu4", "ieen5", "ieet5", "o", "oa", "om", "on", "ong", "oong", "oong1", "u", "ua", "ung", "uo0n", "uo0ng", "uyt1", "uo0t1", "uo0i2", "uo0m2", "uo0n2", "uo0ng2", "uo0i3", "uo0c5", "uo0m5", "uo0t5", "uowc1", "a2", "ai2", "an2", "ang2", "anh2", "ao2", "au2", "ay2", "a1", "ac1", "ach1", "ai1", "am1", "an1", "ang1", "anh1", "ao1", "at1", "au1", "aan", "aang", "aau", "aay", "a4", "ai4", "ang4", "ao4", "e2", "em2", "en2", "eo2", "e1", "ec1", "en1", "eng1", "eo1", "ep1", "et1", "ee", "een", "eenh", "eeu", "i2", "ia2", "im2", "inh2", "iu2", "i1", "ich1", "im1", "inh1", "ip1", "it1", "iu1", "o2", "oi2", "on2", "ong2", "o1", "oc1", "oi1", "on1", "ong1", "op1", "ot1", "o0", "o0i", "o0m", "o0n", "o0ng", "o4", "om4", "ong4", "u2", "ua2", "ui2", "um2", "un2", "ung2", "u1", "ua1", "ui1", "un1", "ung1", "up1", "ut1", "awm", "awn", "awng", "i4", "inh4", "iu4", "ow", "owi", "owm", "own", "uwng", "uwu", "uwowi", "uwowm", "uwown", "uwowng", "uwowu", "uwowc1", "uwowm1", "uwowng1", "uwowp1", "uwowu1", "uwowi3", "a5", "ac5", "ach5", "ai5", "an5", "ang5", "anh5", "ao5", "ap5", "at5", "au5", "a3", "ai3", "an3", "ang3", "anh3", "ao3", "au3", "ay3", "aac1", "aam1", "aan1", "aang1", "aap1", "aat1", "aau1", "aay1", "aam2", "aan2", "aau2", "aay2", "aam3", "aan3", "aau3", "aay3", "aam4", "aan4", "aang4", "aau4", "aay4", "aac5", "aam5", "aan5", "aap5", "aat5", "aau5", "aay5", "awc1", "awn1", "awng1", "awp1", "awt1", "awm2", "awn2", "awng2", "awm3", "awn3", "awm4", "awng4", "awc5", "awm5", "awn5", "awng5", "awp5", "awt5", "e5", "ec5", "en5", "eo5", "ep5", "et5", "e3", "em3", "eo3", "e4", "en4", "eo4", "ee1", "een1", "eep1", "eet1", "ee2", "een2", "eenh2", "eeu2", "ee3", "een3", "ee4", "ee5", "eech5", "een5", "eenh5", "eet5", "eeu5", "i3", "inh3", "iu3", "i5", "ia5", "ich5", "in5", "inh5", "ip5", "it5", "iu5", "o5", "oc5", "on5", "ong5", "ot5", "o3", "oi3", "om3", "ong3", "o01", "o0c1", "o0i1", "o0n1", "o0ng1", "o0p1", "o0t1", "o02", "o0i2", "o0m2", "o0n2", "o0ng2", "o03", "o0i3", "o0n3", "o0ng3", "o04", "o0ng4", "o05", "o0c5", "o0i5", "o0n5", "o0ng5", "o0p5", "o0t5", "ow1", "owi1", "owp1", "owt1", "ow2", "owi2", "owm2", "ow3", "owi3", "ow4", "own4", "ow5", "owm5", "own5", "owp5", "owt5", "u5", "ua5", "uc5", "ui5", "um5", "ung5", "up5", "ut5", "u3", "ua3", "un3", "ung3", "uw1", "uwa1", "uwc1", "uwng1", "uwt1", "uwa2", "uwng2", "uwa3", "uwng3", "uwu3", "uwa4", "uw5", "uwa5", "uwc5", "uwng5", "uwt5", "c", "ch", "im", "iu", "ieeng", "ieeu", "ieep1", "ieeng2", "oai", "oang", "oe", "oen", "oi", "oai2", "oang2", "oa1", "oac1", "oai1", "oan1", "oang1", "oai4", "oen2", "oong2", "oac5", "oai5", "oang5", "oang3", "oawt1", "oet5", "ui", "um", "un", "uy", "uyeen", "uyeen1", "uyeet1", "uyeen2", "uyeen3", "uyeen5", "uo0i", "uo0m", "uaan3", "uee1", "ueenh1", "ueech5", "ueenh5", "uo0c1", "uo0i1", "uo0i4", "uo0i5", "uo0ng5", "am2", "ay1", "aam", "em1", "eem", "in1", "oe2", "om2", "oe1", "uy2", "uc1", "um1", "uy1", "ia4", "in4", "ui4", "um4", "un4", "uw", "uwa", "uwn", "uwowm2", "uwowng2", "uwowng3", "uwowng4", "uwowc5", "uwowp5", "am5", "ay5", "awm1", "awng3", "awn4", "eng3", "eech1", "eem1", "eenh1", "eenh3", "eem4", "eenh4", "eem5", "in3", "oe5", "oi5", "on3", "o0i4", "o0m4", "owm1", "own1", "own2", "owm3", "owi5", "uy3", "uw2", "uw3", "uwi3", "uw4", "uwng4", "uwu5", "ooc1", "uo0n1", "uo0ng1", "uo0m4", "uo0n5", "ap1", "am4", "om1", "oi4", "u4", "ung4", "uwowi1", "uwowi2", "uwowi4", "am3", "om5", "op5", "o0m1", "o0m5", "owi4", "owm4", "ui3", "um3", "uwu1", "uwu2", "uwu4", "d", "ieem2", "ieem4", "ieen4", "ieec5", "ieem5", "ieep5", "ieeu5", "oanh", "oanh2", "oan4", "uyeet5", "uaan4", "uaat5", "ueenh2", "uee5", "an4", "ay4", "ua4", "uwown1", "uwowi5", "uwowng5", "uwowt5", "oa5", "o0m3", "uy5", "g", "gh", "gi", "n", "eeng", "eec1", "eeng1", "eeu1", "eeng2", "eeu4", "in2", "uwowm5", "h", "ieem3", "oan", "oay", "oan2", "oanh1", "oay1", "oang4", "oet1", "oawng", "oach5", "oan5", "oanh5", "oat5", "oai3", "oan3", "oanh3", "oawc1", "oawm1", "oawn2", "oawng2", "oawn3", "oawng4", "oawc5", "oen3", "uynh", "uyeen4", "uaan", "uee", "ueenh", "uych1", "uynh1", "uow", "uaan1", "ueech1", "uee2", "uynh2", "uych5", "y", "anh4", "oa2", "oa4", "y1", "im4", "aang3", "en3", "em4", "oa3", "o0n4", "own3", "uwm2", "y3", "k", "kh", "ieeng4", "ieeng5", "oeo", "oach1", "oat1", "oeo2", "oawn", "oawn1", "oawng1", "oawm2", "oeo3", "uya", "uaang", "uaay", "uyp1", "uaat1", "uaay1", "uee3", "uyu5", "uynh3", "uyu3", "ia1", "uwowt1", "oe3", "ieeng3", "eem2", "y2", "y5", "y4", "l", "uyn", "uaan5", "uo0ng4", "uy4", "uwown2", "uwown5", "em5", "eeu3", "im5", "un5", "m", "uwowu4", "ia3", "im3", "ng", "ngh", "oao", "oam2", "oao1", "oap1", "oeo1", "oam5", "oay5", "oay3", "oawp5", "oawt5", "oeo5", "uaay2", "uaay3", "uaay5", "nh", "uaan2", "uo0m1", "on4", "aang5", "oawm", "oap5", "oawng3", "p", "ph", "eng2", "uwown4", "uwn2", "qu", "ynh", "yeen", "yeen1", "yeet1", "yeen2", "yeen3", "yeen5", "yeet5", "ych1", "ynh1", "yt1", "yu1", "aang2", "ynh2", "yp5", "yt5", "ynh3", "r", "uwowu5", "eep5", "s", "uyt5", "t", "th", "uo0ng3", "uo0n4", "uow3", "oe4", "tr", "uowng1", "uwowu2", "v", "x", "yeem", "yeeng", "yeeu", "yeem1", "yeep1", "yeeu1", "yeem3", "yeeng3", "yeeu3", "dd", "ueenh3"]

phu_nam_dau = {'b':{'rep': 'b', 'vow': 'bê'},
               'c':{'rep': 'c', 'vow': 'xê'},
               'ch':{'rep': 'ch', 'vow': 'chờ'},
               'd':{'rep': 'd', 'vow': 'dê'},
               'đ':{'rep': 'dd', 'vow': 'đê'},
               'g':{'rep': 'g', 'vow': 'gờ'},
               'gh':{'rep': 'gh', 'vow': 'gờ'},
               'gi':{'rep': 'gi', 'vow': 'gi'},
               'h':{'rep': 'h', 'vow': 'hắt'},
               'k':{'rep': 'k', 'vow': 'ka'},
               'kh':{'rep': 'kh', 'vow': 'khờ'},
               'l':{'rep': 'l', 'vow': 'lờ'},
               'm':{'rep': 'm', 'vow': 'mờ'},
               'n':{'rep': 'n', 'vow': 'nờ'},
               'ng':{'rep': 'ng', 'vow': 'ngờ'},
               'ngh':{'rep': 'ngh', 'vow': 'ngờ'},
               'nh':{'rep': 'nh', 'vow': 'nhờ'},
               'p':{'rep': 'p', 'vow': 'pê'},
               'ph':{'rep': 'ph', 'vow': 'phờ'},
               'qu':{'rep': 'qu', 'vow': 'quờ'},
               'r':{'rep': 'r', 'vow': 'rờ'},
               's':{'rep': 's', 'vow': 'ét xì'},
               't':{'rep': 't', 'vow': 'tê'},
               'th':{'rep': 'th', 'vow': 'thờ'},
               'tr':{'rep': 'tr', 'vow': 'trờ'},
               'v':{'rep': 'v', 'vow': 'vê'},
               'x':{'rep': 'x', 'vow': 'ích xì'}}
nguyen_am_don = {
    "a":{"tone":0, "origin": 'a','rep': 'a'},
    'á':{"tone":1, "origin": 'a', 'rep': 'a'},
    'à':{"tone":2, "origin": 'a', 'rep': 'a'},
    'ả':{"tone":3, "origin": 'a', 'rep': 'a'},
    'ã':{"tone":4, "origin": 'a', 'rep': 'a'},
    'ạ':{"tone":5, "origin": 'a', 'rep': 'a'}, 
    "ă":{"tone":0, "origin": 'ă', 'rep': 'aw'},
    'ắ':{"tone":1, "origin": 'ă', 'rep': 'aw'},
    'ằ':{"tone":2, "origin": 'ă', 'rep': 'aw'},
    'ẳ':{"tone":3, "origin": 'ă', 'rep': 'aw'},
    'ẵ':{"tone":4, "origin": 'ă', 'rep': 'aw'},
    'ặ':{"tone":5, "origin": 'ă', 'rep': 'aw'},
    "â":{"tone":0, "origin": 'â', 'rep': 'aa'},
    'ấ':{"tone":1, "origin": 'â', 'rep': 'aa'},
    'ầ':{"tone":2, "origin": 'â', 'rep': 'aa'},
    'ẩ':{"tone":3, "origin": 'â', 'rep': 'aa'},
    'ẫ':{"tone":4, "origin": 'â', 'rep': 'aa'},
    'ậ':{"tone":5, "origin": 'â', 'rep': 'aa'},
    "e":{"tone":0, "origin": 'e', 'rep': 'e'},
    'é':{"tone":1, "origin": 'e', 'rep': 'e'},
    'è':{"tone":2, "origin": 'e', 'rep': 'e'},
    'ẻ':{"tone":3, "origin": 'e', 'rep': 'e'},
    'ẽ':{"tone":4, "origin": 'e', 'rep': 'e'},
    'ẹ':{"tone":5, "origin": 'e', 'rep': 'e'},
    "ê":{"tone":0, "origin": 'ê', 'rep': 'ee'},
    'ế':{"tone":1, "origin": 'ê', 'rep': 'ee'},
    'ề':{"tone":2, "origin": 'ê', 'rep': 'ee'},
    'ể':{"tone":3, "origin": 'ê', 'rep': 'ee'},
    'ễ':{"tone":4, "origin": 'ê', 'rep': 'ee'},
    'ệ':{"tone":5, "origin": 'ê', 'rep': 'ee'},
    "i":{"tone":0, "origin": 'i', 'rep': 'i'},
    'í':{"tone":1, "origin": 'i', 'rep': 'i'},
    'ì':{"tone":2, "origin": 'i', 'rep': 'i'},
    'ỉ':{"tone":3, "origin": 'i', 'rep': 'i'},
    'ĩ':{"tone":4, "origin": 'i', 'rep': 'i'},
    'ị':{"tone":5, "origin": 'i', 'rep': 'i'},
    "o":{"tone":0, "origin": 'o', 'rep': 'o'},
    'ó':{"tone":1, "origin": 'o', 'rep': 'o'},
    'ò':{"tone":2, "origin": 'o', 'rep': 'o'},
    'ỏ':{"tone":3, "origin": 'o', 'rep': 'o'},
    'õ':{"tone":4, "origin": 'o', 'rep': 'o'},
    'ọ':{"tone":5, "origin": 'o', 'rep': 'o'}, 
    "ô":{"tone":0, "origin": 'ô', 'rep': 'o0'},
    'ố':{"tone":1, "origin": 'ô', 'rep': 'o0'},
    'ồ':{"tone":2, "origin": 'ô', 'rep': 'o0'},
    'ổ':{"tone":3, "origin": 'ô', 'rep': 'o0'},
    'ỗ':{"tone":4, "origin": 'ô', 'rep': 'o0'},
    'ộ':{"tone":5, "origin": 'ô', 'rep': 'o0'},
    "ơ":{"tone":0, "origin": 'ơ', 'rep': 'ow'},
    'ớ':{"tone":1, "origin": 'ơ', 'rep': 'ow'},
    'ờ':{"tone":2, "origin": 'ơ', 'rep': 'ow'},
    'ở':{"tone":3, "origin": 'ơ', 'rep': 'ow'},
    'ỡ':{"tone":4, "origin": 'ơ', 'rep': 'ow'},
    'ợ':{"tone":5, "origin": 'ơ', 'rep': 'ow'},
    "u":{"tone":0, "origin": 'u', 'rep': 'u'},
    'ú':{"tone":1, "origin": 'u', 'rep': 'u'},
    'ù':{"tone":2, "origin": 'u', 'rep': 'u'},
    'ủ':{"tone":3, "origin": 'u', 'rep': 'u'},
    'ũ':{"tone":4, "origin": 'u', 'rep': 'u'},
    'ụ':{"tone":5, "origin": 'u', 'rep': 'u'},
    "ư":{"tone":0, "origin": 'ư', 'rep': 'uw'},
    'ứ':{"tone":1, "origin": 'ư', 'rep': 'uw'},
    'ừ':{"tone":2, "origin": 'ư', 'rep': 'uw'},
    'ử':{"tone":3, "origin": 'ư', 'rep': 'uw'},
    'ữ':{"tone":4, "origin": 'ư', 'rep': 'uw'},
    'ự':{"tone":5, "origin": 'ư', 'rep': 'uw'},
    "y":{"tone":0, "origin": 'y', 'rep': 'y'},
    'ý':{"tone":1, "origin": 'y', 'rep': 'y'},
    'ỳ':{"tone":2, "origin": 'y', 'rep': 'y'},
    'ỷ':{"tone":3, "origin": 'y', 'rep': 'y'},
    'ỹ':{"tone":4, "origin": 'y', 'rep': 'y'},
    'ỵ':{"tone":5, "origin": 'y', 'rep': 'y'}
}

phu_am_cuoi = {
    'c': {'rep': 'c'},
    'ch': {'rep': 'ch'},
    'ng': {'rep': 'ng'},
    'nh': {'rep': 'nh'},
    'm': {'rep': 'm'},
    'n': {'rep': 'n'},
    'p': {'rep': 'p'},
    't': {'rep': 't'}, 
    'k': {'rep': 'k'}
}

def _replace_nguyen_am(word):
    tone = 0
    rep = ''
    for c in word:
        if c in nguyen_am_don:
            rep += nguyen_am_don[c]['rep']
            if tone<=0:
                tone = nguyen_am_don[c]['tone']
        else:
            rep += c
    if tone>0:
        rep += str(tone)
    return rep

def parse_word(word, replaced=True, tach_phu_am=False):
    result   = []
    len_word = len(word)
    if len_word>0:
        # xac dinh phu am dau
        if len_word>3:
            if word[:3] in phu_nam_dau:
                result.append(word[:3])
                word = word[3:]
            elif word[:2] in phu_nam_dau:
                result.append(word[:2])
                word = word[2:]
            elif word[0] in phu_nam_dau:
                result.append(word[0])
                word = word[1:]
        elif len_word>2:
            if word[:2] in phu_nam_dau:
                result.append(word[:2])
                word = word[2:]
            elif word[0] in phu_nam_dau:
                result.append(word[0])
                word = word[1:]
        else:
            if word[0] in phu_nam_dau:
                result.append(word[0])
                word = word[1:]

        len_word = len(word)
        if replaced and len(result)>0:
            result[0] = phu_nam_dau[result[0]]['rep']
        #xac  dinh phu am cuoi
        pac = ''
        if tach_phu_am and len_word>0:
            if len_word>2:
                if word[len_word-2:] in phu_am_cuoi:
                    pac = word[len_word-2:]
                    word = word[:len_word-2]
                elif word[-1] in phu_am_cuoi:
                    pac = word[-1]
                    word = word[:len_word-1]
            else:
                if word[-1] in phu_am_cuoi:
                    pac = word[-1]
                    word = word[:len_word-1]
            if pac!='' and tach_phu_am:
                pac=phu_am_cuoi[pac]['rep']
        #xac dinh nguyen am
        len_word = len(word)
        if len_word>0:
            if replaced:
                word = _replace_nguyen_am(word)
            result.append(word)
            if pac!='':
                result.append(pac)
    return result