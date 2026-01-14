#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import curses
import csv
import os
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
from dataclasses import dataclass
from typing import Dict, List, Tuple

DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven"]
LOCKED_DAY_INDEX = 2  # Mercredi
GROUPS = ["G1", "G2", "G3", "G4"]
NB_PERIODE_JOUR = 3  # Nombre de fois ou le bloc GROUPS se repete dans la journee

ATELIERS_CSV = "atelier.csv"
PLANNING_CSV = "planning.csv"


@dataclass
class Atelier:
    id: str
    nom: str
    couleur: str
    designation: str
    extras: Dict[str, str]


def load_ateliers(path: str) -> List[Atelier]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier ateliers introuvable: {path}\n"
            f"Crée un '{path}' avec colonnes: id,nom,designation"
        )
    ateliers: List[Atelier] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            extras = {
                k: v
                for k, v in cleaned.items()
                if k not in {"id", "nom", "designation", "couleur"} and v
            }
            ateliers.append(
                Atelier(
                    id=cleaned.get("id", ""),
                    nom=cleaned.get("nom", ""),
                    couleur=cleaned.get("couleur", ""),
                    designation=cleaned.get("designation", ""),
                    extras=extras,
                )
            )
    # garde seulement ceux qui ont un id
    ateliers = [a for a in ateliers if a.id]
    if not ateliers:
        raise ValueError("Aucun atelier valide trouvé (colonne 'id' vide ?).")
    return ateliers


def empty_planning() -> Dict[str, List[List[str]]]:
    # planning[group][periode] = liste 5 jours -> atelier_id (ou "")
    return {g: [[""] * len(DAYS) for _ in range(NB_PERIODE_JOUR)] for g in GROUPS}


def load_planning(path: str) -> Dict[str, List[List[str]]]:
    if not os.path.exists(path):
        return empty_planning()

    planning = empty_planning()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_period = "periode" in (reader.fieldnames or [])
        for row in reader:
            grp = (row.get("groupe") or "").strip()
            if grp not in planning:
                continue
            if has_period:
                try:
                    periode = int((row.get("periode") or "1").strip() or 1)
                except ValueError:
                    continue
            else:
                periode = 1
            if not (1 <= periode <= NB_PERIODE_JOUR):
                continue
            period_idx = periode - 1
            for i, d in enumerate(DAYS):
                planning[grp][period_idx][i] = (row.get(d) or "").strip()
    return planning


def save_planning(path: str, planning: Dict[str, List[List[str]]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["groupe", "periode"] + DAYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in GROUPS:
            for p in range(NB_PERIODE_JOUR):
                row = {"groupe": g, "periode": str(p + 1)}
                for i, d in enumerate(DAYS):
                    row[d] = planning[g][p][i]
                writer.writerow(row)


def rotate_week(planning: Dict[str, List[List[str]]], direction: int) -> None:
    """
    direction = +1 => décale à droite (Ven->Lun)
    direction = -1 => décale à gauche (Lun->Ven)
    Mercredi (colonne verrouillée) reste à sa place.
    """
    idxs = [0, 1, 3, 4]  # tous sauf Mer (2)
    for g in GROUPS:
        for p in range(NB_PERIODE_JOUR):
            vals = [planning[g][p][i] for i in idxs]
            if direction > 0:
                vals = [vals[-1]] + vals[:-1]
            else:
                vals = vals[1:] + [vals[0]]
            for k, i in enumerate(idxs):
                planning[g][p][i] = vals[k]


def copy_selection_over_workdays(
    planning: Dict[str, List[List[str]]],
    rows: List[Tuple[int, str]],
    sel_i: int,
    sel_j: int,
) -> str:
    """
    Copie la cellule sélectionnée vers les autres jours ouvrés (Lun, Mar, Jeu, Ven),
    en décalant d'un jour et en avançant dans GROUPS à chaque copie. Mercredi est ignoré.
    """
    if not (0 <= sel_i < len(rows)) or not (0 <= sel_j < len(DAYS)):
        return "Sélection invalide."
    if sel_j == LOCKED_DAY_INDEX:
        return "Mercredi est non utilisé, aucune copie effectuée."

    src_p, src_g = rows[sel_i]
    if src_g not in planning or not (0 <= src_p < len(planning[src_g])):
        return "Sélection hors planning."

    src_val = planning[src_g][src_p][sel_j]
    if not src_val:
        return "Cellule vide, rien à copier."

    working_days = [d for d in range(len(DAYS)) if d != LOCKED_DAY_INDEX]
    if sel_j not in working_days:
        return "Jour non utilisable pour la copie."
    try:
        src_group_idx = GROUPS.index(src_g)
    except ValueError:
        return "Groupe inconnu pour la copie."

    start_idx = working_days.index(sel_j)
    for step in range(1, len(working_days)):
        dest_day = working_days[(start_idx + step) % len(working_days)]
        dest_group = GROUPS[(src_group_idx + step) % len(GROUPS)]
        if dest_group not in planning:
            continue
        if not (0 <= src_p < len(planning[dest_group])):
            continue
        planning[dest_group][src_p][dest_day] = src_val

    return "Copie effectuée (+1 jour, groupe suivant, mercredi sauté)."


def copy_all_over_workdays(planning: Dict[str, List[List[str]]], rows: List[Tuple[int, str]]) -> str:
    """
    Applique la logique de copie sur chaque valeur unique rencontrée (hors mercredi),
    en conservant le même décalage (+1 jour, groupe suivant, mercredi sauté).
    """
    working_days = [d for d in range(len(DAYS)) if d != LOCKED_DAY_INDEX]
    first_occurrence_by_value: Dict[str, Tuple[int, int]] = {}
    for i, (p, g) in enumerate(rows):
        grp_rows = planning.get(g)
        if not grp_rows or not (0 <= p < len(grp_rows)):
            continue
        row_vals = grp_rows[p]
        for day in working_days:
            if not (0 <= day < len(row_vals)):
                continue
            val = row_vals[day]
            if val and val not in first_occurrence_by_value:
                first_occurrence_by_value[val] = (i, day)

    if not first_occurrence_by_value:
        return "Aucune cellule non vide à copier."

    applied = 0
    total = len(first_occurrence_by_value)
    # Snapshot des positions à traiter pour ne pas réutiliser les valeurs fraîchement copiées
    for sel_i, sel_j in list(first_occurrence_by_value.values()):
        msg = copy_selection_over_workdays(planning, rows, sel_i, sel_j)
        if msg.startswith("Copie effectuée"):
            applied += 1

    return f"Copie en masse: {applied}/{total} valeurs traitées."


def atelier_label(ateliers_by_id: Dict[str, Atelier], atelier_id: str) -> str:
    if not atelier_id:
        return ""
    a = ateliers_by_id.get(atelier_id)
    if not a:
        return f"?{atelier_id}"
    parts: List[str] = []
    if a.designation:
        parts.append(a.designation)
    for k, v in a.extras.items():
        parts.append(f"{k}: {v}")
    if parts:
        return f"{a.nom}: " + " | ".join(parts)
    return a.nom


def atelier_choice_label(a: Atelier) -> str:
    parts: List[str] = []
    if a.designation:
        parts.append(a.designation)
    for k, v in a.extras.items():
        parts.append(f"{k}: {v}")
    if parts:
        return f"{a.nom} - " + " | ".join(parts)
    return a.nom


def normalize_color_name(name: str) -> str:
    return name.strip().lower()


def color_name_to_curses(name: str) -> Tuple[int, int] | None:
    palette = {
        "noir": curses.COLOR_BLACK,
        "black": curses.COLOR_BLACK,
        "rouge": curses.COLOR_RED,
        "red": curses.COLOR_RED,
        "vert": curses.COLOR_GREEN,
        "green": curses.COLOR_GREEN,
        "jaune": curses.COLOR_YELLOW,
        "yellow": curses.COLOR_YELLOW,
        "bleu": curses.COLOR_BLUE,
        "blue": curses.COLOR_BLUE,
        "magenta": curses.COLOR_MAGENTA,
        "violet": curses.COLOR_MAGENTA,
        "purple": curses.COLOR_MAGENTA,
        "cyan": curses.COLOR_CYAN,
        "blanc": curses.COLOR_WHITE,
        "white": curses.COLOR_WHITE,
    }
    bg = palette.get(name)
    if bg is None:
        return None
    light_bg = bg in (curses.COLOR_YELLOW, curses.COLOR_WHITE, curses.COLOR_GREEN)
    fg = curses.COLOR_BLACK if light_bg else curses.COLOR_WHITE
    return fg, bg


def export_pdf(
    path: str, planning: Dict[str, List[List[str]]], ateliers_by_id: Dict[str, Atelier]
) -> None:
    def pdf_color_from_name(name: str) -> colors.Color | None:
        palette = {
            "noir": colors.Color(0.1, 0.1, 0.1),
            "black": colors.Color(0.1, 0.1, 0.1),
            "rouge": colors.Color(0.9, 0.3, 0.3),
            "red": colors.Color(0.9, 0.3, 0.3),
            "vert": colors.Color(0.6, 0.9, 0.6),
            "green": colors.Color(0.6, 0.9, 0.6),
            "jaune": colors.Color(0.97, 0.9, 0.5),
            "yellow": colors.Color(0.97, 0.9, 0.5),
            "bleu": colors.Color(0.5, 0.7, 0.95),
            "blue": colors.Color(0.5, 0.7, 0.95),
            "magenta": colors.Color(0.85, 0.6, 0.9),
            "violet": colors.Color(0.85, 0.6, 0.9),
            "purple": colors.Color(0.85, 0.6, 0.9),
            "cyan": colors.Color(0.6, 0.9, 0.95),
            "blanc": colors.Color(0.95, 0.95, 0.95),
            "white": colors.Color(0.95, 0.95, 0.95),
        }
        return palette.get(name)

    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"]
    cell_style.fontName = "Helvetica"
    cell_style.fontSize = 8
    cell_style.leading = 10

    header_style = styles["Heading4"]
    header_style.fontName = "Helvetica-Bold"
    header_style.fontSize = 9
    header_style.leading = 11

    data: List[List[Paragraph]] = []
    data.append(
        [Paragraph("Groupe", header_style)] + [Paragraph(d, header_style) for d in DAYS]
    )
    for p in range(NB_PERIODE_JOUR):
        for g in GROUPS:
            row_label = f"{g} Periode {p + 1}" if NB_PERIODE_JOUR > 1 else g
            row: List[Paragraph] = [Paragraph(row_label, cell_style)]
            for j in range(len(DAYS)):
                if j == LOCKED_DAY_INDEX:
                    row.append(Paragraph("", cell_style))
                    continue
                atelier_id = planning[g][p][j]
                label = atelier_label(ateliers_by_id, atelier_id)
                row.append(Paragraph(label or "", cell_style))
            data.append(row)

    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        path,
        pagesize=page_size,
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    usable_width = page_size[0] - doc.leftMargin - doc.rightMargin
    col_width = usable_width / (len(DAYS) + 1)
    table = Table(data, colWidths=[col_width] * (len(DAYS) + 1), repeatRows=1)

    table_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ]
    )

    row_idx = 1
    for p in range(NB_PERIODE_JOUR):
        for g in GROUPS:
            for col_idx, _ in enumerate(DAYS, start=1):
                if col_idx - 1 == LOCKED_DAY_INDEX:
                    continue
                atelier_id = planning[g][p][col_idx - 1]
                if not atelier_id:
                    continue
                a = ateliers_by_id.get(atelier_id)
                if not a or not a.couleur:
                    continue
                bg = pdf_color_from_name(normalize_color_name(a.couleur))
                if not bg:
                    continue
                luminance = 0.299 * bg.red + 0.587 * bg.green + 0.114 * bg.blue
                fg = colors.white if luminance < 0.55 else colors.black
                table_style.add(
                    "BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), bg
                )
                table_style.add("TEXTCOLOR", (col_idx, row_idx), (col_idx, row_idx), fg)
            row_idx += 1

    table.setStyle(table_style)
    doc.build([table])


def pick_atelier(stdscr, ateliers: List[Atelier], current_id: str) -> str:
    """
    Mini sélection (liste) : ↑/↓, Entrée = choisir, Esc = annuler, Backspace = vider
    """
    curses.curs_set(0)
    h, w = stdscr.getmaxyx()
    win_h = min(18, h - 4)
    win_w = min(70, w - 4)
    y0 = (h - win_h) // 2
    x0 = (w - win_w) // 2
    win = curses.newwin(win_h, win_w, y0, x0)
    win.keypad(True)

    # index par défaut
    idx = 0
    for i, a in enumerate(ateliers):
        if a.id == current_id:
            idx = i
            break

    top = 0

    while True:
        win.erase()
        win.box()
        title = " Choisir un atelier (↑/↓, Entrée=OK, Esc=Annuler, Backspace=Vider) "
        win.addstr(0, max(1, (win_w - len(title)) // 2), title)

        visible = win_h - 2
        if idx < top:
            top = idx
        if idx >= top + visible:
            top = idx - visible + 1

        for row in range(visible):
            i = top + row
            if i >= len(ateliers):
                break
            a = ateliers[i]
            line = f"{a.id:>3}  {atelier_choice_label(a)}"
            line = line[: win_w - 4]
            if i == idx:
                win.attron(curses.A_REVERSE)
                win.addstr(1 + row, 2, line)
                win.attroff(curses.A_REVERSE)
            else:
                win.addstr(1 + row, 2, line)

        win.refresh()
        ch = win.getch()

        if ch in (27,):  # ESC
            return current_id
        if ch in (curses.KEY_ENTER, 10, 13):
            return ateliers[idx].id
        if ch in (curses.KEY_UP,):
            idx = max(0, idx - 1)
        elif ch in (curses.KEY_DOWN,):
            idx = min(len(ateliers) - 1, idx + 1)
        elif ch in (curses.KEY_NPAGE,):
            idx = min(len(ateliers) - 1, idx + 10)
        elif ch in (curses.KEY_PPAGE,):
            idx = max(0, idx - 10)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            return ""


def draw(
    stdscr,
    planning: Dict[str, List[List[str]]],
    ateliers_by_id: Dict[str, Atelier],
    sel: Tuple[int, int],
    status: str,
):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # En-tête
    header = "Agenda console (Lun–Ven) — Mercredi verrouillé"
    help1 = "Fleches: bouger | Entree: choisir atelier | N: reinitialiser | S: sauver | R: recharger | P: pdf | C: copier cellule selectionnee (+1 jour, groupe suivant, mercredi saute) | V: copier 1 cellule par jour (meme logique) | <: rot gauche | >: rot droite | Q: quitter"
    stdscr.addstr(0, 2, header[: w - 4], curses.A_BOLD)
    stdscr.addstr(1, 2, help1[: w - 4])

    # Calcul dimensions
    top_y = 3
    left_x = 2
    max_label_len = max(
        (len(atelier_label(ateliers_by_id, a.id)) for a in ateliers_by_id.values()),
        default=0,
    )
    cell_w = max(
        12, min(max(24, max_label_len + 2), (w - left_x - 6) // (len(DAYS) + 1))
    )

    def wrap_label(label: str, width: int) -> List[str]:
        if not label:
            return [""]
        return textwrap.wrap(label, width=width, break_long_words=True) or [""]

    rows = [(p, g) for p in range(NB_PERIODE_JOUR) for g in GROUPS]
    row_heights: List[int] = []
    for p, g in rows:
        max_lines = 1
        for j in range(len(DAYS)):
            atelier_id = planning[g][p][j]
            label = "" if j == LOCKED_DAY_INDEX else atelier_label(ateliers_by_id, atelier_id)
            max_lines = max(max_lines, len(wrap_label(label, cell_w - 3)))
        row_heights.append(max_lines)

    grid_w = (len(DAYS) + 1) * cell_w + 1
    separator_count = max(0, NB_PERIODE_JOUR - 1)
    grid_h = sum(row_heights) + len(rows) + 4 + separator_count

    if top_y + grid_h + 4 > h or left_x + grid_w > w:
        stdscr.addstr(
            top_y, 2, "Fenêtre trop petite. Agrandis le terminal.", curses.A_BOLD
        )
        stdscr.refresh()
        return

    # Couleurs groupes
    # (fallback si terminal sans couleurs)
    use_color = curses.has_colors()
    if use_color:
        curses.start_color()
        curses.use_default_colors()
        # paires: 1..4
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)  # mercredi verrouill‚
        color_pairs: Dict[str, int] = {}
        next_pair = 10
        for a in ateliers_by_id.values():
            key = normalize_color_name(a.couleur)
            if not key or key in color_pairs:
                continue
            fg_bg = color_name_to_curses(key)
            if not fg_bg:
                continue
            fg, bg = fg_bg
            curses.init_pair(next_pair, fg, bg)
            color_pairs[key] = next_pair
            next_pair += 1
    else:
        color_pairs = {}

    def box(y, x, hh, ww):
        for i in range(ww):
            stdscr.addch(y, x + i, curses.ACS_HLINE)
            stdscr.addch(y + hh - 1, x + i, curses.ACS_HLINE)
        for i in range(hh):
            stdscr.addch(y + i, x, curses.ACS_VLINE)
            stdscr.addch(y + i, x + ww - 1, curses.ACS_VLINE)
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, x + ww - 1, curses.ACS_URCORNER)
        stdscr.addch(y + hh - 1, x, curses.ACS_LLCORNER)
        stdscr.addch(y + hh - 1, x + ww - 1, curses.ACS_LRCORNER)

    # Dessin en-têtes colonnes
    box(top_y, left_x, grid_h, grid_w)
    stdscr.addstr(top_y + 1, left_x + 2, "Groupe".ljust(cell_w - 3), curses.A_BOLD)
    for j, d in enumerate(DAYS):
        x = left_x + (j + 1) * cell_w
        stdscr.addstr(top_y + 1, x + 2, d.ljust(cell_w - 3), curses.A_BOLD)

    # Séparateurs verticaux
    for j in range(1, len(DAYS) + 1):
        x = left_x + j * cell_w
        for yy in range(top_y + 1, top_y + grid_h - 1):
            stdscr.addch(yy, x, curses.ACS_VLINE)

    y = top_y + 2
    for xx in range(left_x + 1, left_x + grid_w - 1):
        stdscr.addch(y, xx, curses.ACS_HLINE)

    sel_i, sel_j = sel  # i: row index, j: jour index
    y = top_y + 3
    for row_idx, (p, g) in enumerate(rows):
        row_height = row_heights[row_idx]
        color_attr = curses.color_pair((row_idx % len(GROUPS)) + 1) if use_color else 0
        group_label = f"{g} Periode {p + 1}" if NB_PERIODE_JOUR > 1 else g
        group_lines = wrap_label(group_label, cell_w - 3)
        day_cells: List[Tuple[List[str], int]] = []
        for j in range(len(DAYS)):
            atelier_id = planning[g][p][j]
            label = "" if j == LOCKED_DAY_INDEX else atelier_label(ateliers_by_id, atelier_id)
            lines = wrap_label(label, cell_w - 3)
            cell_attr = color_attr
            if use_color and atelier_id and j != LOCKED_DAY_INDEX:
                a = ateliers_by_id.get(atelier_id)
                if a and a.couleur:
                    key = normalize_color_name(a.couleur)
                    pair_id = color_pairs.get(key)
                    if pair_id:
                        cell_attr = curses.color_pair(pair_id)
            day_cells.append((lines, cell_attr))

        for line_idx in range(row_height):
            group_text = group_lines[line_idx] if line_idx < len(group_lines) else ""
            stdscr.addstr(
                y + line_idx,
                left_x + 2,
                group_text.ljust(cell_w - 3),
                color_attr | curses.A_BOLD,
            )

            for j in range(len(DAYS)):
                x = left_x + (j + 1) * cell_w
                locked = j == LOCKED_DAY_INDEX
                locked_attr = curses.color_pair(5) if (use_color and locked) else 0
                is_sel = row_idx == sel_i and j == sel_j
                cell_lines, cell_attr = day_cells[j]
                base_attr = cell_attr
                if locked:
                    base_attr = locked_attr | curses.A_DIM
                if is_sel:
                    base_attr |= curses.A_REVERSE

                text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                stdscr.addstr(y + line_idx, x + 2, text.ljust(cell_w - 3), base_attr)

        y += row_height
        for xx in range(left_x + 1, left_x + grid_w - 1):
            stdscr.addch(y, xx, curses.ACS_HLINE)
        y += 1

        is_period_end = (row_idx + 1) % len(GROUPS) == 0
        if is_period_end and p < NB_PERIODE_JOUR - 1:
            for xx in range(left_x + 1, left_x + grid_w - 1):
                stdscr.addch(y, xx, curses.ACS_HLINE)
            sep_label = f"Periode {p + 2}"
            stdscr.addstr(y, left_x + 2, sep_label[: cell_w - 3], curses.A_BOLD)
            y += 1
    # Panneau détails atelier sélectionné
    sel_p, sel_g = rows[sel_i]
    aid = planning[sel_g][sel_p][sel_j]
    a = ateliers_by_id.get(aid) if sel_j != LOCKED_DAY_INDEX else None
    details_y = top_y + grid_h + 1
    stdscr.addstr(details_y, 2, "Détails:", curses.A_BOLD)
    if sel_j != LOCKED_DAY_INDEX:
        if a:
            label_prefix = f"{sel_g} Periode {sel_p + 1} - " if NB_PERIODE_JOUR > 1 else ""
            detail_lines = [f"{label_prefix}{a.id} - {a.nom}"]
            if a.designation:
                detail_lines.append(a.designation)
            if a.couleur:
                detail_lines.append(f"couleur: {a.couleur}")
            for k, v in a.extras.items():
                detail_lines.append(f"{k}: {v}")
            for i, line in enumerate(detail_lines):
                y_line = details_y + 1 + i
                if y_line >= h - 1:
                    break
                stdscr.addstr(y_line, 4, line[: max(0, w - 5)])
        else:
            if details_y + 1 < h - 1:
                msg = "(vide)" if not aid else f"Atelier inconnu: {aid}"
                stdscr.addstr(details_y + 1, 4, msg[: max(0, w - 5)])

    # Status
    stdscr.addstr(h - 2, 2, (status or "").ljust(w - 4)[: w - 4], curses.A_DIM)
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)

    ateliers = load_ateliers(ATELIERS_CSV)
    ateliers_by_id = {a.id: a for a in ateliers}
    planning = load_planning(PLANNING_CSV)
    rows = [(p, g) for p in range(NB_PERIODE_JOUR) for g in GROUPS]

    sel_i, sel_j = 0, 0
    status = "OK"

    while True:
        draw(stdscr, planning, ateliers_by_id, (sel_i, sel_j), status)
        status = ""

        ch = stdscr.getch()

        if ch in (ord("q"), ord("Q")):
            break

        if ch == curses.KEY_UP:
            sel_i = max(0, sel_i - 1)
        elif ch == curses.KEY_DOWN:
            sel_i = min(len(rows) - 1, sel_i + 1)
        elif ch == curses.KEY_LEFT:
            sel_j = max(0, sel_j - 1)
        elif ch == curses.KEY_RIGHT:
            sel_j = min(len(DAYS) - 1, sel_j + 1)

        elif ch in (curses.KEY_ENTER, 10, 13):
            # Editer cellule si pas mercredi
            if sel_j == LOCKED_DAY_INDEX:
                status = "Mercredi est verrouillé (non éditable)."
                continue
            p, g = rows[sel_i]
            current_id = planning[g][p][sel_j]
            new_id = pick_atelier(stdscr, ateliers, current_id)
            planning[g][p][sel_j] = new_id
            status = "Cellule mise à jour."

        elif ch in (ord("s"), ord("S")):
            try:
                save_planning(PLANNING_CSV, planning)
                status = f"Saved to {PLANNING_CSV}."
            except Exception as exc:
                status = f"Save failed: {exc}"
        elif ch in (ord("r"), ord("R")):
            try:
                planning = load_planning(PLANNING_CSV)
                status = f"Reloaded {PLANNING_CSV}."
            except Exception as exc:
                status = f"Reload failed: {exc}"
        elif ch in (ord("n"), ord("N")):
            planning = empty_planning()
            status = "Planning reinitialise."
        elif ch in (ord("p"), ord("P")):
            try:
                export_pdf("planning.pdf", planning, ateliers_by_id)
                status = "Exported planning.pdf."
            except Exception as exc:
                status = f"PDF export failed: {exc}"
        elif ch == ord("<"):
            rotate_week(planning, -1)
            status = "Rotated left."
        elif ch == ord(">"):
            rotate_week(planning, 1)
            status = "Rotated right."
        elif ch in (ord("c"), ord("C")):
            status = copy_selection_over_workdays(planning, rows, sel_i, sel_j)
        elif ch in (ord("v"), ord("V")):
            status = copy_all_over_workdays(planning, rows)


if __name__ == "__main__":
    curses.wrapper(main)
