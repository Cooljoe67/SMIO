import pytest

from scripts import import_labelled_tsv
from src.db.models import Email

HEADER = (
    "SCHLUESSEL\tsplit\tVORGANG\tdirection\tsenderName\tsenderAddress\trecipientAddress\t"
    "subject\tbody\tattachmentNames\tisOwnDomainSender\tisReply\tisForward\tGEPRUEFT\t"
    "DIFFICULTY\tFOLDER\tSTATUS\tPRIORITY\tACTION\tMAIL_TYPE_1\tMAIL_TYPE_2\tNOTE\tTRAITS\t"
    "VORSCHLAG_FOLDER\tVORSCHLAG_STATUS\tVORSCHLAG_PRIORITY\tVORSCHLAG_ACTION\t"
    "VORSCHLAG_MAIL_TYPE_1\tVORSCHLAG_MAIL_TYPE_2\tVORSCHLAG_BEGRUENDUNG"
)
LABELLED = (
    "Z-1\tTRAIN\tVG:NOR-1\tINBOUND\tSea Freight\tops@sea.de\tinfo@rc.de\tDGD Container\t"
    "Hallo,\\nanbei die DGD.\tDGD.pdf\tnein\tnein\tnein\t\tMEDIUM\t"
    "Dokumente & Frachtpapiere\tZu erledigen\tNormal\tBestätigung erforderlich\tDokument\t"
    "Rückfrage\t\t\tDokumente & Frachtpapiere\tZu erledigen\tNormal\t"
    "Bestätigung erforderlich\tDokument\tRückfrage\tGefahrgutpapiere zur Prüfung."
)
UNLABELLED = (
    "Z-2\tTEST\t\tINTERNAL\tK. Meier\tk.meier@rc.de\talle@rc.de\tTeam\tInfo\t\tja\tnein\tnein"
    "\t\tEASY" + "\t" * 15
)
OTHER = LABELLED.replace("Z-1", "Z-3").replace("Dokumente & Frachtpapiere", "Sonstiges")


def _write(tmp_path, *rows):
    path = tmp_path / "data.tsv"
    path.write_text("﻿" + "\n".join((HEADER, *rows)) + "\n", encoding="utf-8")
    return path


def test_rows_are_imported_as_labelled_training_data(db, tmp_path):
    path = _write(tmp_path, LABELLED, UNLABELLED, OTHER)

    assert import_labelled_tsv.import_tsv(db, path, "logistics") == {
        "created": 3, "updated": 0, "labelled": 2,
    }

    labelled, unlabelled, other = db.query(Email).order_by(Email.dataset_key).all()
    assert labelled.folder_list == "logistics"
    assert labelled.true_label == labelled.classification == "dokumente"
    assert labelled.classification_source == "dataset"
    assert (labelled.true_status, labelled.true_mail_type_2) == ("Zu erledigen", "Rückfrage")
    assert labelled.classification_reason == "Gefahrgutpapiere zur Prüfung."
    assert labelled.text == "Hallo,\nanbei die DGD."
    assert labelled.case_ref == "VG:NOR-1"
    assert (labelled.is_own_domain_sender, labelled.reviewed) == (False, None)
    assert labelled.is_training_data and labelled.processed == 1
    assert labelled.account is None

    assert unlabelled.dataset_split == "TEST"
    assert unlabelled.true_label is None and unlabelled.classification is None
    assert unlabelled.is_own_domain_sender is True
    assert other.true_label == "other"


def test_reimport_updates_rows_instead_of_duplicating(db, tmp_path):
    import_labelled_tsv.import_tsv(db, _write(tmp_path, LABELLED), "logistics")
    changed = LABELLED.replace("Zu erledigen\tNormal\tBestätigung", "Abgeschlossen\tNormal\tBestätigung", 1)

    result = import_labelled_tsv.import_tsv(db, _write(tmp_path, changed), "logistics")

    assert result == {"created": 0, "updated": 1, "labelled": 1}
    assert db.query(Email).one().true_status == "Abgeschlossen"


def test_unknown_folder_is_rejected(db, tmp_path):
    path = _write(tmp_path, LABELLED.replace("Dokumente & Frachtpapiere", "Unbekannt", 1))

    with pytest.raises(ValueError, match="Unbekannt"):
        import_labelled_tsv.import_tsv(db, path, "logistics")


def test_real_dataset_folders_all_map_to_the_folder_list():
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "t1a-arbeitsblatt.tsv"
    if not path.exists():
        pytest.skip("dataset not present")
    labels = import_labelled_tsv.folder_labels("logistics")
    with open(path, encoding="utf-8-sig", newline="") as file:
        names = {row["FOLDER"] for row in csv.DictReader(file, delimiter="\t") if row["FOLDER"]}

    assert names <= set(labels)


def test_own_company_is_replaced_with_a_neutral_one(db, tmp_path):
    own = (
        "Z-4\tTRAIN\t\tINTERNAL\tNord Spedition\tk.meier@nord-spedition.de\talle@nord-spedition.de\t"
        "Info von nord-spedition.de\tGrüße, Ihr Nord-Spedition Team aus Nord\\nNord Spedition GmbH\tNORD_SPEDITION.pdf\tja\tnein\tnein"
        "\t\tEASY" + "\t" * 15
    )
    inbound = LABELLED.replace("info@rc.de", "dispo@nord-spedition.de")

    import_labelled_tsv.import_tsv(db, _write(tmp_path, own, inbound), "logistics")

    internal, received = db.query(Email).order_by(Email.dataset_key.desc()).all()
    assert (internal.sender_name, internal.sender) == ("Firma", "k.meier@firma.example")
    assert internal.recipients == "alle@firma.example"
    assert internal.subject == "Info von firma.example"
    # The full name is replaced; a single word of it (often a place) is kept.
    assert internal.text == "Grüße, Ihr Firma Team aus Nord\nFirma GmbH"
    assert internal.attachment_names == "Firma.pdf"
    assert received.recipients == "dispo@firma.example"


def test_keep_company_leaves_the_data_unchanged(db, tmp_path):
    own = UNLABELLED.replace("k.meier@rc.de", "k.meier@nord-spedition.de")

    import_labelled_tsv.import_tsv(db, _write(tmp_path, own), "logistics", keep_company=True)

    assert db.query(Email).one().sender == "k.meier@nord-spedition.de"


def test_a_one_word_company_name_is_not_replaced_in_the_text():
    neutralize = import_labelled_tsv.neutralizer("rc.de")

    assert neutralize("Sendung RC-2026-0902 an info@rc.de") == "Sendung RC-2026-0902 an info@firma.example"
