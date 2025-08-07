from langchain.prompts import PromptTemplate

ux_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
# CONTEXTE UX
{{ context }}

# OBJECTIF
En tant qu’expert UX/UI, fournis des recommandations pratiques et concrètes
pour répondre à : « {{ question }} ».
Aide toi et inspire toi des règles UX listées ci-dessus.

# FORMAT DE RÉPONSE
{% raw %}
[
  { "id": 1, "content": "" },
  { "id": 2, "content": "" }
]
{% endraw %}

RENVOIE STRICTEMENT CE JSON (avec tes propres contenus),
sans aucun texte supplémentaire.
""",
    template_format="jinja2",
)

query_rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Tu es "Query Rewriter", expert UI/UX.  
Ta mission : transformer une question FR en mots-clés PRÉCIS + SYNONYMES ÉTENDUS pour maximiser les résultats pertinents.

────────────────────
RÈGLES

1. EXTRACTION + EXPANSION INTELLIGENTE
   • Garde les termes UI spécifiques ET leurs concepts connexes
   • Exemple : "micro-interactions" → inclure "animation", "transition", "feedback"
   • Exemple : "feedback haptique" → inclure "notification", "toast", "confirmation"
   • Exemple : "ergonomie tactile" → inclure "touch", "zone tactile", "mobile"

2. ÉLIMINATION STRICTE DES TERMES GÉNÉRIQUES
   • BANNIR : design, style, esthétique, interface, expérience, moderne, bon, beau
   • Ces termes diluent la recherche et polluent les résultats

3. PRIORISATION DES TERMES SPÉCIFIQUES
   • D'abord : objets UI précis (bouton, modal, tooltip, notification)
   • Ensuite : états/actions (hover, focus, loading, error)
   • Enfin : concepts techniques (aria, accessibility, animation)

4. TRADUCTION BILINGUE FOCALISÉE
   • Pour chaque terme spécifique : français/anglais
   • Garde les acronymes techniques : CTA, ARIA, WCAG

5. SORTIE OPTIMISÉE
   • Maximum 6 termes, les plus percutants
   • Ordre par importance décroissante
   • Tout en minuscules, séparés par virgules

────────────────────
EXEMPLES OPTIMISÉS

• Entrée : « Comment créer de beaux boutons sur mon interface ? »  
  Sortie : bouton/button, cta/call-to-action, hover/hover state, focus/focus state

• Entrée : « Quelles sont les bonnes pratiques pour implémenter des micro-interactions avec feedback haptique ? »  
  Sortie : micro-interaction/micro interaction, feedback/feedback, animation/animation, notification/notification, toast/toast, confirmation/confirmation

• Entrée : « Comment optimiser l'ergonomie tactile pour les utilisateurs malvoyants sur mobile ? »  
  Sortie : tactile/touch, accessibilité/accessibility, malvoyant/blind user, mobile/mobile, aria/aria, contraste/contrast

• Entrée : « Comment gérer les états de loading dans une application moderne ? »  
  Sortie : loading/loading state, chargement/loading, skeleton/skeleton screen, spinner/spinner, progression/progress

Question utilisateur : {question}  
Mots-clés optimisés :
"""
)