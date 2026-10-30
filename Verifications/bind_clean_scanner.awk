#!/usr/bin/awk -f
# DATA file: helper scanner for bind_clean_public_api.sh.
# The verify target glob only executes *.sh, so this .awk is data, not an entry point.
#
# Scans a single C++ header and reports public parameters that are not
# pybind11-friendly: non-const lvalue references to fundamental / fixed-width
# alias types, plus mutable pointers of any type. Non-const references to
# class types are permitted.
#
# Public scope means namespace scope (free functions are public API) plus the
# public sections of classes/structs. Protected/private sections, function
# bodies, constructor initializer lists, and lambda bodies are ignored.
#
# Module/ModuleStream.h is intentionally excluded: streaming operators take
# std::ostream& by design and are not intended to be exposed as value-in/value-out
# Python bindings.
#
# Residual simplifications:
# - Nested local classes inside function or lambda bodies are skipped along with
#   the body and therefore not diagnosed.
# - A parameter whose first non-qualifier token is a lowercase unqualified
#   non-alias type may be missed; the scanner treats such segments as probable
#   expressions to avoid false positives on parenthesized initializers.

function is_word_char(c) {
    return (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9") || c == "_"
}

function preprocess(line,    i, c, out) {
    out = ""
    i = 1
    # If a string/char literal continued from the previous line, a pending
    # backslash escape consumed the line break; reset it here.
    if (in_string || in_char) {
        string_esc = 0
    }
    while (i <= length(line)) {
        c = substr(line, i, 1)
        if (in_ml_comment) {
            if (c == "*" && substr(line, i + 1, 1) == "/") {
                out = out "  "
                i += 2
                in_ml_comment = 0
            } else {
                out = out " "
                i++
            }
            continue
        }
        if (in_string) {
            if (string_esc) {
                out = out " "
                string_esc = 0
                i++
            } else if (c == "\\") {
                out = out " "
                string_esc = 1
                i++
            } else if (c == "\"") {
                out = out "\""
                in_string = 0
                i++
            } else {
                out = out " "
                i++
            }
            continue
        }
        if (in_char) {
            if (string_esc) {
                out = out " "
                string_esc = 0
                i++
            } else if (c == "\\") {
                out = out " "
                string_esc = 1
                i++
            } else if (c == "'") {
                out = out "'"
                in_char = 0
                i++
            } else {
                out = out " "
                i++
            }
            continue
        }
        if (c == "/" && substr(line, i + 1, 1) == "/") {
            while (i <= length(line)) {
                out = out " "
                i++
            }
            break
        }
        if (c == "/" && substr(line, i + 1, 1) == "*") {
            in_ml_comment = 1
            out = out "  "
            i += 2
            continue
        }
        if (c == "\"") {
            in_string = 1
            out = out "\""
            i++
            continue
        }
        if (c == "'") {
            in_char = 1
            out = out "'"
            i++
            continue
        }
        out = out c
        i++
    }
    return out
}

function line_of(big, pos,    n, i) {
    n = 1
    for (i = 1; i < pos; i++) {
        if (substr(big, i, 1) == "\n") {
            n++
        }
    }
    return n
}

function word_before(big, pos,    j, end, start) {
    j = pos - 1
    while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
        j--
    }
    if (j < 1) {
        return ""
    }
    end = j
    while (j >= 1 && is_word_char(substr(big, j, 1))) {
        j--
    }
    start = j + 1
    return substr(big, start, end - start + 1)
}

function word_before_pos(big, pos,    j, end, start) {
    j = pos - 1
    while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
        j--
    }
    if (j < 1) {
        return ""
    }
    end = j
    while (j >= 1 && is_word_char(substr(big, j, 1))) {
        j--
    }
    start = j + 1
    return substr(big, start, end - start + 1)
}

function is_denylisted_keyword(s) {
    return (s ~ /^(static_assert|sizeof|decltype|alignas|if|for|while|switch|catch|return|noexcept|requires|typeid|throw)$/)
}

function is_declarator_before(big, pos,    w, j, ch, end, start) {
    j = pos - 1
    while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
        j--
    }
    if (j < 1) {
        return 0
    }
    ch = substr(big, j, 1)
    if (ch == "~") {
        return 1
    }
    # operator(): the param list '(' follows the symbol '()'.
    if (ch == ")") {
        j--
        while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
            j--
        }
        if (j >= 1 && substr(big, j, 1) == "(") {
            j--
            while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
                j--
            }
            if (j >= 1) {
                end = j
                while (j >= 1 && is_word_char(substr(big, j, 1))) {
                    j--
                }
                start = j + 1
                w = substr(big, start, end - start + 1)
                if (w == "operator") {
                    return 1
                }
            }
        }
    }
    # operator[], operator new[], operator delete[]: the param list '('
    # follows the symbol '[]'.
    if (ch == "]") {
        j--
        while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
            j--
        }
        if (j >= 1 && substr(big, j, 1) == "[") {
            j--
            while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
                j--
            }
            if (j >= 1) {
                end = j
                while (j >= 1 && is_word_char(substr(big, j, 1))) {
                    j--
                }
                start = j + 1
                w = substr(big, start, end - start + 1)
                if (w == "operator" || w == "new" || w == "delete") {
                    return 1
                }
            }
        }
    }
    # Symbolic operators like operator=, operator+, operator->*.
    if (ch ~ /[=+\-*/%^&|~!<>,.:]/) {
        k = j
        while (k >= 1 && substr(big, k, 1) ~ /[=+\-*/%^&|~!<>,.:]/) {
            k--
        }
        while (k >= 1 && substr(big, k, 1) ~ /[ \t\n]/) {
            k--
        }
        if (k >= 1) {
            end = k
            while (k >= 1 && is_word_char(substr(big, k, 1))) {
                k--
            }
            start = k + 1
            w = substr(big, start, end - start + 1)
            if (w == "operator") {
                return 1
            }
        }
    }
    w = word_before(big, pos)
    if (w == "") {
        return 0
    }
    if (is_denylisted_keyword(w)) {
        return 0
    }
    if (w == "operator") {
        return 1
    }
    return 1
}

function is_body_preceding_keyword(w) {
    return (w ~ /^(mutable|noexcept|const|volatile|override|final|requires|class|struct|union|enum|typename|static|inline|virtual|explicit|constexpr|friend|public|protected|private|return|if|else|while|for|do|switch|case|default|catch|try|throw|new|delete|using|namespace|template|concept|co_await|co_return|co_yield|auto|decltype|void|bool|char|int|unsigned|long|short|signed|float|double|size_t|int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|float16_t|float32_t|float64_t|bfloat16_t|int4x2_t)$/)
}

function is_operator_bracket(big, close_pos,    open_pos, k, end, start, w) {
    # Return 1 if the ']' at close_pos belongs to operator[], operator new[],
    # or operator delete[].
    if (close_pos <= 1) {
        return 0
    }
    open_pos = close_pos - 1
    while (open_pos >= 1 && substr(big, open_pos, 1) ~ /[ \t\n]/) {
        open_pos--
    }
    if (open_pos < 1 || substr(big, open_pos, 1) != "[") {
        return 0
    }
    k = open_pos - 1
    while (k >= 1 && substr(big, k, 1) ~ /[ \t\n]/) {
        k--
    }
    if (k < 1) {
        return 0
    }
    end = k
    while (k >= 1 && is_word_char(substr(big, k, 1))) {
        k--
    }
    start = k + 1
    w = substr(big, start, end - start + 1)
    if (w == "operator" || w == "new" || w == "delete") {
        return 1
    }
    return 0
}

function find_body_brace(big, start_pos,    q, ch, pd, ad, bd, w) {
    pd = 0
    ad = 0
    bd = 0
    q = start_pos
    while (q <= length(big)) {
        ch = substr(big, q, 1)
        if (ch == "(") {
            pd++
        } else if (ch == ")") {
            if (pd > 0) {
                pd--
            }
        } else if (ch == "<") {
            ad++
        } else if (ch == ">") {
            if (ad > 0) {
                ad--
            }
        } else if (ch == "{") {
            if (bd > 0) {
                bd++
            } else if (pd == 0 && ad == 0) {
                w = word_before_pos(big, q)
                if (w == "" || is_body_preceding_keyword(w)) {
                    return q
                }
                bd++
            }
        } else if (ch == "}") {
            if (bd > 0) {
                bd--
            }
        }
        q++
    }
    return 0
}

function split_params(s, arr,    n, i, pd, ad, ch, buf) {
    n = 0
    pd = 0
    ad = 0
    buf = ""
    for (i = 1; i <= length(s); i++) {
        ch = substr(s, i, 1)
        if (ch == "(") {
            pd++
        } else if (ch == ")") {
            pd--
        } else if (ch == "<") {
            ad++
        } else if (ch == ">") {
            if (ad > 0) {
                ad--
            }
        }
        if (ch == "," && pd == 0 && ad == 0) {
            arr[++n] = buf
            buf = ""
        } else {
            buf = buf ch
        }
    }
    arr[++n] = buf
    return n
}

function is_token_at(s, pos, token,    len) {
    len = length(token)
    if (substr(s, pos, len) != token) {
        return 0
    }
    if (pos > 1 && is_word_char(substr(s, pos - 1, 1))) {
        return 0
    }
    if (pos + len <= length(s) && is_word_char(substr(s, pos + len, 1))) {
        return 0
    }
    return 1
}

function access_label_advance(big, pos,    len, i) {
    if (is_token_at(big, pos, "public")) {
        len = 6
    } else if (is_token_at(big, pos, "protected")) {
        len = 9
    } else if (is_token_at(big, pos, "private")) {
        len = 7
    } else {
        return 0
    }
    i = pos + len
    while (i <= length(big) && substr(big, i, 1) ~ /[ \t]/) {
        i++
    }
    if (i > length(big) || substr(big, i, 1) != ":") {
        return 0
    }
    if (i + 1 <= length(big) && substr(big, i + 1, 1) == ":") {
        return 0
    }
    return i - pos + 1
}

function strip_default_arg(seg,    i, pd, ad, ch, out) {
    pd = 0
    ad = 0
    out = ""
    for (i = 1; i <= length(seg); i++) {
        ch = substr(seg, i, 1)
        if (ch == "(") {
            pd++
        } else if (ch == ")") {
            pd--
        } else if (ch == "<") {
            ad++
        } else if (ch == ">") {
            if (ad > 0) {
                ad--
            }
        }
        if (ch == "=" && pd == 0 && ad == 0) {
            break
        }
        out = out ch
    }
    return out
}

function mask_angle_brackets(seg,    i, ch, out, ad) {
    ad = 0
    out = ""
    for (i = 1; i <= length(seg); i++) {
        ch = substr(seg, i, 1)
        if (ch == "<") {
            ad++
            out = out "<"
        } else if (ch == ">") {
            if (ad > 0) {
                ad--
            }
            out = out ">"
        } else if (ad > 0) {
            out = out " "
        } else {
            out = out ch
        }
    }
    return out
}

function first_type_token(s,    i, ch, token) {
    i = 1
    while (i <= length(s) && substr(s, i, 1) ~ /[ \t]/) {
        i++
    }
    while (i <= length(s)) {
        if (is_token_at(s, i, "const")) {
            i += 5
        } else if (is_token_at(s, i, "volatile")) {
            i += 8
        } else if (is_token_at(s, i, "typename")) {
            i += 8
        } else {
            break
        }
        while (i <= length(s) && substr(s, i, 1) ~ /[ \t]/) {
            i++
        }
    }
    token = ""
    while (i <= length(s)) {
        ch = substr(s, i, 1)
        if (ch ~ /[ \t,\&\*]/) {
            break
        }
        token = token ch
        i++
    }
    return token
}

function is_type_like(tok) {
    if (tok == "") {
        return 0
    }
    if (tok ~ /^decltype\(/) {
        return 1
    }
    if (tok ~ /::/) {
        return 1
    }
    if (tok ~ /</) {
        return 1
    }
    if (tok ~ /_t$/) {
        return 1
    }
    if (tok ~ /^[A-Z]/) {
        return 1
    }
    if (tok ~ /x2_t$/) {
        return 1
    }
    if (tok ~ /^(void|bool|char|auto|decltype|unsigned|long|short|signed|int|float|double|size_t|int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|float16_t|float32_t|float64_t|bfloat16_t|int4x2_t)$/) {
        return 1
    }
    return 0
}

function is_banned_fundamental(tok) {
    # Closed list of C++ fundamentals and the project's fixed-width aliases,
    # including packed dtypes (x2/packed forms). Must stay in sync with Tensor/DTypes.h.
    return (tok ~ /^(bool|char|short|int|long|float|double|size_t|int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|float16_t|float16x2_t|float32_t|float64_t|bfloat16_t|bfloat16x2_t|int4x2_t)$/)
}

function check_segment(seg, file, ln,    masked, type_tok, i, j, ch, pre, has_star, has_const, after_const) {
    seg = strip_default_arg(seg)
    gsub(/^[ \t]+|[ \t]+$/, "", seg)
    if (seg == "") {
        return 0
    }

    masked = mask_angle_brackets(seg)
    type_tok = first_type_token(masked)
    if (!is_type_like(type_tok)) {
        return 0
    }

    # Non-const lvalue reference to a fundamental / fixed-width-alias type.
    for (i = 1; i <= length(masked); i++) {
        ch = substr(masked, i, 1)
        if (ch == "&") {
            if (substr(masked, i + 1, 1) == "&") {
                i++
                continue
            }
            pre = substr(masked, 1, i - 1)
            has_star = 0
            for (j = 1; j <= length(pre); j++) {
                if (substr(pre, j, 1) == "*") {
                    has_star = 1
                    break
                }
            }
            has_const = 0
            if (!has_star) {
                j = 1
                while (j <= length(pre) - 4) {
                    if (is_token_at(pre, j, "const")) {
                        has_const = 1
                        break
                    }
                    j++
                }
            }
            if (has_star || !has_const) {
                if (!is_banned_fundamental(type_tok)) {
                    continue
                }
                printf("%s:%d: non-const lvalue reference to fundamental type: %s\n", file, ln, seg) > "/dev/stderr"
                return 1
            }
        }
    }

    # Mutable pointer: a * whose pointee is not const and which is not itself
    # const (* const).
    for (i = 1; i <= length(masked); i++) {
        ch = substr(masked, i, 1)
        if (ch == "*") {
            after_const = 0
            j = i + 1
            while (j <= length(masked) && substr(masked, j, 1) ~ /[ \t]/) {
                j++
            }
            if (is_token_at(masked, j, "const")) {
                after_const = 1
            }
            if (after_const) {
                continue
            }

            has_const = 0
            j = i - 1
            while (j >= 1 && substr(masked, j, 1) != "*") {
                if (j >= 5 && is_token_at(masked, j - 4, "const")) {
                    has_const = 1
                    break
                }
                j--
            }
            if (has_const) {
                continue
            }

            printf("%s:%d: mutable pointer parameter: %s\n", file, ln, seg) > "/dev/stderr"
            return 1
        }
    }

    return 0
}

BEGIN {
    in_ml_comment = 0
    in_string = 0
    in_char = 0
    string_esc = 0
}

{
    cleaned = preprocess($0)
    if (NR > 1) {
        big = big "\n"
    }
    big = big cleaned
}

END {
    file = FILENAME
    n = length(big)
    pos = 1
    depth = 0
    paren_depth = 0
    angle_depth = 0
    class_count = 0
    pending_class = ""
    body_skip = 0
    body_target_depth = 0
    found_violation = 0

    while (pos <= n) {
        c = substr(big, pos, 1)

        if (body_skip) {
            if (c == "{") {
                depth++
            } else if (c == "}") {
                if (depth > 0) {
                    depth--
                }
                if (depth <= body_target_depth) {
                    body_skip = 0
                }
            }
            pos++
            continue
        }

        if (c == "<") {
            angle_depth++
        } else if (c == ">") {
            if (angle_depth > 0) {
                angle_depth--
            }
        }

        if (angle_depth == 0 && c == "c" && substr(big, pos, 5) == "class" && (pos == 1 || !is_word_char(substr(big, pos - 1, 1))) && !is_word_char(substr(big, pos + 5, 1))) {
            pending_class = "class"
            pos += 5
            continue
        }
        if (angle_depth == 0 && c == "s" && substr(big, pos, 6) == "struct" && (pos == 1 || !is_word_char(substr(big, pos - 1, 1))) && !is_word_char(substr(big, pos + 6, 1))) {
            pending_class = "struct"
            pos += 6
            continue
        }

        if (class_count > 0 && depth == class_start[class_count] + 1) {
            adv = access_label_advance(big, pos)
            if (adv > 0) {
                if (is_token_at(big, pos, "public")) {
                    class_access[class_count] = "public"
                } else if (is_token_at(big, pos, "protected")) {
                    class_access[class_count] = "protected"
                } else {
                    class_access[class_count] = "private"
                }
                pos += adv
                continue
            }
        }

        if (c == "(") {
            checked = (class_count == 0) || (depth == class_start[class_count] + 1 && class_access[class_count] == "public")
            if (checked && paren_depth == 0 && angle_depth == 0) {
                # Lambda: [...](...){...}
                j = pos - 1
                while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
                    j--
                }
                if (j >= 1 && substr(big, j, 1) == "]" && !is_operator_bracket(big, j)) {
                    start_paren = pos
                    pd = 1
                    pbuf = "("
                    pos++
                    while (pos <= n && pd > 0) {
                        ch = substr(big, pos, 1)
                        pbuf = pbuf ch
                        if (ch == "(") {
                            pd++
                        } else if (ch == ")") {
                            pd--
                        }
                        pos++
                    }
                    body_pos = find_body_brace(big, pos)
                    if (body_pos > 0) {
                        pos = body_pos + 1
                        depth++
                        body_skip = 1
                        body_target_depth = depth - 1
                    }
                    continue
                }

                # operator(): the '(' directly after 'operator' is the operator
                # symbol itself, not the parameter list. Consume the empty '()'
                # and let the following '(' be processed as the real param list.
                w = word_before(big, pos)
                if (w == "operator") {
                    j = pos - 1
                    while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
                        j--
                    }
                    if (j >= 1 && substr(big, j, 1) == "r") {
                        pd = 1
                        pos++
                        while (pos <= n && pd > 0) {
                            ch = substr(big, pos, 1)
                            if (ch == "(") {
                                pd++
                            } else if (ch == ")") {
                                pd--
                            }
                            pos++
                        }
                        continue
                    }
                }

                if (is_declarator_before(big, pos)) {
                    start_paren = pos
                    pd = 1
                    pbuf = "("
                    pos++
                    while (pos <= n && pd > 0) {
                        ch = substr(big, pos, 1)
                        pbuf = pbuf ch
                        if (ch == "(") {
                            pd++
                        } else if (ch == ")") {
                            pd--
                        }
                        pos++
                    }
                    inner = substr(pbuf, 2, length(pbuf) - 2)
                    ln = line_of(big, start_paren)

                    q = pos
                    while (q <= n && substr(big, q, 1) ~ /[ \t\n]/) {
                        q++
                    }
                    confirmed = 0
                    if (q <= n) {
                        rest = substr(big, q, 200)
                        if (rest ~ /^(;|\{|:|const[^A-Za-z0-9_]|noexcept[^A-Za-z0-9_]|override[^A-Za-z0-9_]|final[^A-Za-z0-9_]|volatile[^A-Za-z0-9_]|requires[^A-Za-z0-9_]|=|->|&|&&)/) {
                            confirmed = 1
                        }
                    }

                    if (confirmed) {
                        np = split_params(inner, params)
                        for (pi = 1; pi <= np; pi++) {
                            if (check_segment(params[pi], file, ln)) {
                                found_violation = 1
                            }
                        }
                        body_pos = find_body_brace(big, pos)
                        if (body_pos > 0) {
                            pos = body_pos + 1
                            depth++
                            body_skip = 1
                            body_target_depth = depth - 1
                        }
                    }
                    continue
                }
            }
            paren_depth++
            pos++
            continue
        }

        if (c == ")") {
            if (paren_depth > 0) {
                paren_depth--
            }
            pos++
            continue
        }

        if (c == "{") {
            if (paren_depth > 0 || angle_depth > 0) {
                pos++
                continue
            }
            # Lambda without a parameter list: []{...}
            j = pos - 1
            while (j >= 1 && substr(big, j, 1) ~ /[ \t\n]/) {
                j--
            }
            if (j >= 1 && substr(big, j, 1) == "]" && !is_operator_bracket(big, j)) {
                depth++
                body_skip = 1
                body_target_depth = depth - 1
                pos++
                continue
            }
            if (pending_class != "" && (class_count == 0 || depth == class_start[class_count] + 1)) {
                class_count++
                class_start[class_count] = depth
                class_default[class_count] = (pending_class == "struct") ? "public" : "private"
                class_access[class_count] = class_default[class_count]
            }
            pending_class = ""
            depth++
            pos++
            continue
        }

        if (c == "}") {
            if (paren_depth > 0 || angle_depth > 0) {
                pos++
                continue
            }
            if (depth > 0) {
                depth--
            }
            if (class_count > 0 && depth == class_start[class_count]) {
                class_count--
            }
            pos++
            continue
        }

        if (c == ";" && pending_class != "") {
            pending_class = ""
        }

        pos++
    }

    if (found_violation) {
        exit 1
    }
}
