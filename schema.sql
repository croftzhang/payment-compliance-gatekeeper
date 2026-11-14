-- Google Cloud Spanner Property Graph Schema for Payment Compliance Gatekeeper

-- 1. Entities Node Table (Corporates & Individuals)
CREATE TABLE Entities (
    entity_id STRING(64) NOT NULL,
    entity_name STRING(256) NOT NULL,
    entity_type STRING(32) NOT NULL, -- 'CORPORATE' | 'INDIVIDUAL'
    jurisdiction STRING(8) NOT NULL,
    is_sanctioned BOOL NOT NULL,
    sanction_list STRING(64),        -- e.g., 'OFAC-SDN', 'EU-CONS', NULL
    risk_score FLOAT64 NOT NULL
) PRIMARY KEY (entity_id);

-- 2. EntityOwnership Edge Table (owner -> target)
CREATE TABLE EntityOwnership (
    owner_id STRING(64) NOT NULL,
    target_id STRING(64) NOT NULL,
    ownership_pct FLOAT64 NOT NULL,  -- 0.0 ~ 1.0 (e.g., 0.60 = 60%)
    effective_date DATE NOT NULL,
    CONSTRAINT FK_Owner FOREIGN KEY (owner_id) REFERENCES Entities (entity_id),
    CONSTRAINT FK_Target FOREIGN KEY (target_id) REFERENCES Entities (entity_id)
) PRIMARY KEY (owner_id, target_id);

-- 3. Spanner Property Graph Definition
CREATE OR REPLACE PROPERTY GRAPH ComplianceGraph
    NODE TABLES (
        Entities
            KEY (entity_id)
            LABEL Entity PROPERTIES (
                entity_id, entity_name, entity_type,
                jurisdiction, is_sanctioned, sanction_list, risk_score
            )
    )
    EDGE TABLES (
        EntityOwnership
            KEY (owner_id, target_id)
            SOURCE KEY (owner_id) REFERENCES Entities (entity_id)
            DESTINATION KEY (target_id) REFERENCES Entities (entity_id)
            LABEL Owns PROPERTIES (ownership_pct, effective_date)
    );
